from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Neo4j connection
neo4j_config = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "your_password"
}

# Initialize Neo4j driver
driver = GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"]))

# Initialize OpenAI model
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Define schema for the knowledge graph
node_types = ["Project", "Document", "Chunk", "Entity"]
relationship_types = ["HAS_DOCUMENT", "HAS_CHUNK", "MENTIONS"]
patterns = [
    ("Project", "HAS_DOCUMENT", "Document"),
    ("Document", "HAS_CHUNK", "Chunk"),
    ("Chunk", "MENTIONS", "Entity")
]

# Initialize the SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=OpenAIEmbeddings(),
    schema={"node_types": node_types, "relationship_types": relationship_types, "patterns": patterns},
    from_pdf=True
)

# Define the PDF file path
pdf_file_path = "path/to/your/document.pdf"

# Run the pipeline
await kg_builder.run_async(file_path=pdf_file_path)

# Close the Neo4j driver connection
driver.close()

print("Knowledge graph populated in Neo4j successfully!")
