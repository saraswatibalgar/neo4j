import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 1) Load & validate environment
# -----------------------------
load_dotenv()

# Azure OpenAI
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Neo4j
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Optional: control destructive reset (defaults True to mirror original)
RESET_DB = os.getenv("RESET_DB", "true").lower() in {"1", "true", "yes", "y"}

missing = [k for k, v in {
    "AZURE_OPENAI_API_KEY": azure_api_key,
    "AZURE_OPENAI_ENDPOINT": azure_endpoint,
    "OPENAI_API_VERSION": api_version,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": azure_deployment,
    "NEO4J_URI": neo4j_uri,
    "NEO4J_USERNAME": neo4j_user,
    "NEO4J_PASSWORD": neo4j_password,
}.items() if not v]
if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# ----------------------------------------
# 2) Prompt (examples use single-brace Cypher)
# ----------------------------------------
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
1. Analyze the text to identify all entities that match the schema.
2. Generate a `MERGE` query for each unique entity. Use the specified unique property.
   - Example: `MERGE (p:Persona {name: "Content Moderator"})`
3. Generate a `MERGE` query for each relationship identified between the entities.
   - Example: `MERGE (f:Feature {name: "Content Moderation"})-[:HAS_REQUIREMENT]->(r:Requirement {id: "REQ-078"})`
4. Add the source document relationship for every extracted entity.
   - Example: `MERGE (p:Persona {name: "Content Moderator"}) MERGE (d:SourceDocument {name: "prd_v2.docx"}) MERGE (p)-[:MENTIONED_IN]->(d)`
5. Return **only** a list of Cypher queries separated by a semicolon (;). Do not add any other text, explanations, or formatting.

---
Now, process the following text from the document named **{document_name}**:
{input}
"""

# ---------------------
# 3) Utility sanitizers
# ---------------------
CODEBLOCK_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", flags=re.S)
MULTISPACE_RE = re.compile(r"\s+", flags=re.S)


def extract_queries(raw: str) -> list:
    """Extract semicolon-separated Cypher statements from LLM output.
    Handles code fences, stray commentary, and double-brace artifacts.
    """
    if not raw:
        return []

    # If there are fenced blocks, take their contents; otherwise use raw
    blocks = CODEBLOCK_RE.findall(raw)
    text = "\n".join(blocks) if blocks else raw

    # Strip leading 'cypher' tokens or commentary prefixes
    text = text.strip()
    if text.lower().startswith("cypher"):
        text = text[len("cypher"):].strip()

    # Normalize accidental double braces from templating examples
    text = text.replace("{{", "{").replace("}}", "}")

    # Remove accidental markdown bullets like leading dashes before MERGE
    # and collapse excessive whitespace
    text = MULTISPACE_RE.sub(" ", text)

    # Split into statements
    parts = [p.strip() for p in text.split(";")]
    queries = [p for p in parts if p]
    return queries


# ---------------------
# 4) Main pipeline
# ---------------------

def main():
    # LLM (use correct param name: openai_api_key)
    llm = AzureChatOpenAI(
        openai_api_version=api_version,
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_template(CYPHER_GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))

    def setup_schema(sess):
        if RESET_DB:
            print("Resetting database (DETACH DELETE n)...")
            sess.run("MATCH (n) DETACH DELETE n")
        print("Ensuring constraints...")
        constraints = {
            "Persona": "name",
            "Feature": "name",
            "Goal": "action",
            "Objective": "benefit",
            "Requirement": "id",
            "BusinessRule": "rule_id",
            "DataEntity": "name",
            "SourceDocument": "name",
        }
        for label, prop in constraints.items():
            sess.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")

    with driver.session(database=NEO4J_DATABASE) as session:
        setup_schema(session)

    print("Loading documents from 'documents'...")
    loader = DirectoryLoader(
        "documents",
        glob="**/*.*",
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        print("No documents found in 'documents'. Add files and rerun.")
        driver.close()
        return

    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=128)
    chunks = text_splitter.split_documents(docs)
    if not chunks:
        print("No non-empty chunks extracted.")
        driver.close()
        return

    print(f"Processing {len(chunks)} chunks...")

    def run_queries_tx(tx, statements: list):
        for q in statements:
            tx.run(q)

    for i, chunk in enumerate(chunks, start=1):
        document_name = os.path.basename(chunk.metadata.get("source", "Unknown Document"))
        print(f"Chunk {i}/{len(chunks)}: {document_name}")

        try:
            cypher_out = chain.invoke({
                "input": chunk.page_content,
                "document_name": document_name,
            })

            queries = extract_queries(cypher_out)
            if not queries:
                print("  -> No queries returned; skipping.")
                continue

            with driver.session(database=NEO4J_DATABASE) as session:
                # Execute the entire chunk in a single write transaction for atomicity
                session.execute_write(run_queries_tx, queries)
            print(f"  -> Executed {len(queries)} queries.")

        except Exception as e:
            print(f"  !! Error on chunk {i}: {e}")

    driver.close()
    print("Done. Explore the graph in Neo4j Browser.")


if __name__ == "__main__":
    main()
