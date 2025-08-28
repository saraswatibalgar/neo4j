from simplekg import SimpleKGPipeline
from langchain.document_loaders import PyPDFLoader

# Neo4j config
neo4j_config = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "your_password"
}

# Initialize pipeline
pipeline = SimpleKGPipeline(neo4j_config=neo4j_config)

# Example project
project_name = "Project_X"
pdf_paths = ["BRD.pdf", "PRD.pdf", "QA.pdf"]

for pdf_path in pdf_paths:
    doc_type = pdf_path.split("/")[-1].split(".")[0]
    
    # Load PDF and split into chunks
    pages = PyPDFLoader(pdf_path).load_and_split()
    
    # Feed each chunk to the pipeline with metadata
    for i, chunk in enumerate(pages):
        pipeline.add_document(
            text=chunk.page_content,
            metadata={
                "project": project_name,
                "document": doc_type,
                "chunk_index": i
            }
        )

# Build KG in Neo4j
pipeline.build()
print("Neo4j populated successfully!")
