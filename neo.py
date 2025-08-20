import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Load Environment Variables ---
# Securely loads credentials from the .env file.
load_dotenv()

# Azure OpenAI credentials
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Neo4j credentials
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
# Explicitly define the database name for AuraDB
NEO4J_DATABASE = "neo4j"


# --- 2. Define the LLM Prompt for Cypher Generation ---
# This is the core instruction for the AI. It includes the full, generalized
# schema and instructs the model to generate Neo4j MERGE queries directly.
# NOTE: Words like 'name' and 'id' are escaped with double curly braces {{name}}
# to prevent the template parser from treating them as input variables.

CYPHER_GENERATION_PROMPT = """
You are an expert project management analyst and a Neo4j Cypher query expert.
Your task is to extract entities and relationships from the provided text based on the
detailed schema below and generate a list of Cypher `MERGE` queries to create them in a Neo4j database.

**Schema:**
- **Nodes (Entities):**
  - `Persona`: A user, actor, or stakeholder. (Unique Property: `{{name}}`)
  - `Feature`: A high-level piece of functionality or a module. (Unique Property: `{{name}}`)
  - `Goal`: A specific action or capability a Persona wants. (Unique Property: `{{action}}`)
  - `Objective`: The business or user benefit. (Unique Property: `{{benefit}}`)
  - `Requirement`: A specific, testable condition the system must meet. (Unique Property: `{{id}}`)
  - `BusinessRule`: A constraint, policy, or business logic. (Unique Property: `{{rule_id}}`)
  - `DataEntity`: A key data object or concept. (Unique Property: `{{name}}`)
  - `SourceDocument`: The origin document for traceability. (Unique Property: `{{name}}`)

- **Relationships (Edges):**
  - `(Persona)-[:WANTS_TO]->(Goal)`
  - `(Goal)-[:TO_ACHIEVE]->(Objective)`
  - `(Feature)-[:HAS_REQUIREMENT]->(Requirement)`
  - `(Feature)-[:INCLUDES]->(Goal)`
  - `(Feature)-[:CONSTRAINED_BY]->(BusinessRule)`
  - `(Feature)-[:MANAGES]->(DataEntity)`
  - `(Requirement)-[:DERIVED_FROM]->(Objective)`
  - `(Any Node)-[:MENTIONED_IN]->(SourceDocument)`

**Instructions:**
1.  Analyze the text to identify all entities that match the schema.
2.  Generate a `MERGE` query for each unique entity. Use the specified unique property.
    - Example: `MERGE (p:Persona {{name: "Content Moderator"}})`
3.  Generate a `MERGE` query for each relationship identified between the entities.
    - Example: `MERGE (f:Feature {{name: "Content Moderation"}})-[:HAS_REQUIREMENT]->(r:Requirement {{id: "REQ-078"}})`
4.  Add the source document relationship for every extracted entity.
    - Example: `MERGE (p:Persona {{name: "Content Moderator"}}) MERGE (d:SourceDocument {{name: "prd_v2.docx"}}) MERGE (p)-[:MENTIONED_IN]->(d)`
5.  Return **only** a list of Cypher queries separated by a semicolon (;). Do not add any other text, explanations, or formatting.

---
Now, process the following text from the document named **{document_name}**:
{input}
"""

# --- 3. Main Script Logic ---

def main():
    """Main function to run the document extraction and graph import process."""

    # Initialize the Azure OpenAI LLM for reliable, structured output
    llm = AzureChatOpenAI(
        openai_api_version=api_version,
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        temperature=0.0, # Zero temperature for maximum determinism
    )

    # Create the LangChain chain to process text and get Cypher queries
    prompt = ChatPromptTemplate.from_template(CYPHER_GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    # Connect to the Neo4j database
    print("Connecting to Neo4j database...")
    driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))

    # Setup database constraints for data integrity
    with driver.session(database=NEO4J_DATABASE) as session:
        print("Clearing database for a fresh import...")
        session.run("MATCH (n) DETACH DELETE n")

        print("Setting up unique constraints on nodes...")
        constraints = {
            "Persona": "name", "Feature": "name", "Goal": "action",
            "Objective": "benefit", "Requirement": "id", "BusinessRule": "rule_id",
            "DataEntity": "name", "SourceDocument": "name"
        }
        for label, prop in constraints.items():
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")

    # Load all documents from the 'documents' directory using UnstructuredLoader
    print("Loading documents from the 'documents' folder...")
    # DirectoryLoader will automatically use UnstructuredLoader for various file types
    loader = DirectoryLoader("documents", glob="**/*.*", show_progress=True, use_multithreading=True)
    docs = loader.load()

    if not docs:
        print("No documents found. Please add your project files to the 'documents' folder.")
        driver.close()
        return

    # Split documents into manageable chunks for the LLM
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=128)
    chunks = text_splitter.split_documents(docs)
    print(f"Split documents into {len(chunks)} chunks for processing.")

    # Process each chunk and import the generated graph data into Neo4j
    max_retries = 3
    for i, chunk in enumerate(chunks):
        document_name = chunk.metadata.get('source', 'Unknown Document').split(os.sep)[-1]
        print(f"\n--- Processing chunk {i+1}/{len(chunks)} from '{document_name}' ---")

        for attempt in range(max_retries):
            try:
                # Invoke the chain to get the Cypher queries from the LLM
                cypher_queries_str = chain.invoke({
                    "input": chunk.page_content,
                    "document_name": document_name
                })

                # --- FIX: Clean the LLM output before processing ---
                # This removes common artifacts like markdown code blocks or leading text
                cleaned_str = cypher_queries_str.strip()
                if cleaned_str.startswith("```cypher"):
                    cleaned_str = cleaned_str[len("```cypher"):].strip()
                if cleaned_str.startswith("cypher"):
                    cleaned_str = cleaned_str[len("cypher"):].strip()
                if cleaned_str.endswith("```"):
                    cleaned_str = cleaned_str[:-3].strip()


                # Split the string of queries into a list of individual queries
                queries = [q.strip() for q in cleaned_str.split(';') if q.strip()]

                if not queries:
                    print("LLM returned no queries for this chunk.")
                    break # Don't retry if there's nothing to do

                print(f"Generated {len(queries)} Cypher queries.")

                # Execute each query in a transaction to build the graph
                with driver.session(database=NEO4J_DATABASE) as session:
                    for query in queries:
                        session.run(query)

                print("Successfully imported chunk into Neo4j.")
                break # Exit the retry loop on success

            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt + 1 == max_retries:
                    print(f"Failed to process chunk {i+1} after {max_retries} attempts.")
                else:
                    print("Retrying in 3 seconds...")
                    time.sleep(3)

    # Clean up and close the database connection
    driver.close()
    print("\n--- Pipeline Complete ---")
    print("The knowledge graph has been successfully built in your Neo4j Aura database.")
    print("You can now explore it using the Neo4j Browser.")

if __name__ == "__main__":
    main()
