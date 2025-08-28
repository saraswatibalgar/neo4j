from simplekg import SimpleKGBuilder
from langchain.document_loaders import PyPDFLoader

# Neo4j connection
neo4j_config = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "your_password"
}

# Initialize SimpleKGBuilder
kg_builder = SimpleKGBuilder(neo4j_config=neo4j_config)

# Example project
project_name = "Project_X"

# List of document paths
pdf_paths = [
    "path/to/BRD.pdf",
    "path/to/PRD.pdf",
    "path/to/QA.pdf"
]

for pdf_path in pdf_paths:
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()  # returns chunks
    
    # Use file name as document type
    doc_type = pdf_path.split("/")[-1].split(".")[0]
    
    # Add document node
    doc_node = kg_builder.add_node(label="Document", properties={"name": doc_type})
    
    # Add project node if not exists
    proj_node = kg_builder.add_node(label="Project", properties={"name": project_name})
    kg_builder.add_relationship(proj_node, "HAS_DOCUMENT", doc_node)
    
    # Add chunks
    for i, chunk in enumerate(pages):
        chunk_node = kg_builder.add_node(label="Chunk", properties={"text": chunk.page_content, "index": i})
        kg_builder.add_relationship(doc_node, "HAS_CHUNK", chunk_node)

print("Data populated in Neo4j successfully!")
