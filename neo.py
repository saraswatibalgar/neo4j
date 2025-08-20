import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from neo4j import GraphDatabase, basic_auth

# --- 1. Load Environment Variables and Configuration ---
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

# --- 2. Define the Prompt for the LLM ---
# This prompt instructs the LLM to act as an extractor and
# directly generate the Cypher queries needed to build the graph.

CYPHER_GENERATION_PROMPT = """
You are an expert project management analyst and a Neo4j Cypher query expert.
Your task is to extract entities and relationships from the provided text and
generate a list of Cypher `MERGE` queries to create them in a Neo4j database.

**Schema & Rules:**
- **Nodes:**
  - `Persona`: {name: string}
  - `Feature`: {name: string}
  - `Goal`: {action: string}
  - `Objective`: {benefit: string}
- **Relationships:**
  - `(Persona)-[:WANTS_TO]->(Goal)`
  - `(Feature)-[:INCLUDES]->(Goal)`
  - `(Goal)-[:TO_ACHIEVE]->(Objective)`

**Instructions:**
1.  For each entity found in the text, generate a `MERGE` query to create the node.
    - Example Node: `MERGE (p:Persona {name: "Registered User"})`
2.  For each relationship found, generate a `MERGE` query to create the relationship between the nodes.
    - Example Relationship: `MERGE (p:Persona {name: "Registered User"})-[:WANTS_TO]->(g:Goal {action: "upload a profile picture"})`
3.  Return **only** a list of Cypher queries separated by a semicolon (;). Do not add any other text, explanations, or formatting.

**Example Input Text:**
"The User Profile feature allows Registered Users to upload a profile picture so they can personalize their account."

**Example Output:**
MERGE (f:Feature {name: "User Profile"});
MERGE (p:Persona {name: "Registered User"});
MERGE (g:Goal {action: "upload a profile picture"});
MERGE (o:Objective {benefit: "personalize their account"});
MERGE (f)-[:INCLUDES]->(g);
MERGE (p)-[:WANTS_TO]->(g);
MERGE (g)-[:TO_ACHIEVE]->(o);

---
Now, process the following text:
{input}
"""

# --- 3. Main Script Logic ---

def main():
    """Main function to run the extraction and import process."""

    # Initialize the Azure OpenAI LLM
    llm = AzureChatOpenAI(
        openai_api_version=api_version,
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        temperature=0.0 # Set to 0 for deterministic, precise query generation
    )

    # Create the LangChain prompt and chain
    prompt = ChatPromptTemplate.from_template(CYPHER_GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    # Connect to Neo4j
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))

    # Setup database: Clear existing data and create constraints
    with driver.session() as session:
        print("Clearing the database...")
        session.run("MATCH (n) DETACH DELETE n")
        
        print("Setting up constraints...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Persona) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Feature) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Goal) REQUIRE n.action IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Objective) REQUIRE n.benefit IS UNIQUE")

    # Load documents from the 'documents' folder
    print("Loading documents...")
    loader = DirectoryLoader("documents", glob="**/*.*", show_progress=True)
    docs = loader.load()
    if not docs:
        print("No documents found in the 'documents' folder. Exiting.")
        driver.close()
        return

    # Split documents into smaller chunks
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"Split documents into {len(chunks)} chunks.")

    # Process each chunk and import into Neo4j
    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing chunk {i+1}/{len(chunks)} ---")
        
        try:
            # Get the Cypher queries from the LLM
            cypher_queries_str = chain.invoke({"input": chunk.page_content})
            
            # Split the string of queries into a list
            queries = [q.strip() for q in cypher_queries_str.split(';') if q.strip()]

            if not queries:
                print("LLM returned no queries for this chunk.")
                continue

            print(f"Generated {len(queries)} Cypher queries.")

            # Execute each query to build the graph
            with driver.session() as session:
                for query in queries:
                    session.run(query)
            
            print("Successfully imported chunk into Neo4j.")

        except Exception as e:
            print(f"An error occurred while processing chunk {i+1}: {e}")

    # Clean up
    driver.close()
    print("\n--- Pipeline Complete ---")
    print("Knowledge graph has been built in your Neo4j Aura database.")

if __name__ == "__main__":
    main()
