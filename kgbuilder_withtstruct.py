import os
from pathlib import Path
import PyPDF2
from neo4j import GraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

class SimplePDFToNeo4j:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, openai_key):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.setup_database()
    
    def setup_database(self):
        """Create basic constraints and vector index"""
        with self.driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            
            # Create vector index for embeddings
            try:
                session.run("""
                    CALL db.index.vector.createNodeIndex(
                        'chunk_embeddings', 'Chunk', 'embedding', 1536, 'cosine'
                    )
                """)
            except:
                pass  # Index might already exist
    
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def process_pdf(self, pdf_path, project_name, doc_type, doc_name=None):
        """Main function to process PDF and store in Neo4j"""
        
        # Get document name
        if not doc_name:
            doc_name = Path(pdf_path).stem
        
        print(f"Processing {pdf_path}...")
        
        # Extract and chunk text
        text = self.extract_pdf_text(pdf_path)
        chunks = self.splitter.split_text(text)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embeddings.embed_documents(chunks)
        
        # Create unique IDs
        doc_id = f"{project_name}_{doc_name}_{doc_type}".replace(" ", "_")
        
        # Store in Neo4j
        with self.driver.session() as session:
            # Create project
            session.run("""
                MERGE (p:Project {name: $project_name})
                SET p.updated_at = datetime()
            """, project_name=project_name)
            
            # Create document
            session.run("""
                MATCH (p:Project {name: $project_name})
                MERGE (d:Document {id: $doc_id})
                SET d.name = $doc_name,
                    d.type = $doc_type,
                    d.path = $pdf_path,
                    d.chunk_count = $chunk_count
                MERGE (p)-[:HAS_DOCUMENT]->(d)
            """, 
            project_name=project_name, doc_id=doc_id, doc_name=doc_name, 
            doc_type=doc_type, pdf_path=pdf_path, chunk_count=len(chunks))
            
            # Delete old chunks if document exists
            session.run("""
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE c
            """, doc_id=doc_id)
            
            # Create chunks
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                session.run("""
                    MATCH (d:Document {id: $doc_id})
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        text: $text,
                        chunk_index: $index,
                        embedding: $embedding
                    })
                    CREATE (d)-[:HAS_CHUNK]->(c)
                """, doc_id=doc_id, chunk_id=chunk_id, text=chunk_text, 
                index=i, embedding=embedding)
        
        print(f"✅ Successfully processed {doc_name} with {len(chunks)} chunks")
        return {"document": doc_name, "chunks": len(chunks), "status": "success"}
    
    def process_multiple_pdfs(self, project_name, pdf_list):
        """Process multiple PDFs for a project"""
        results = []
        
        for pdf_info in pdf_list:
            try:
                result = self.process_pdf(
                    pdf_path=pdf_info['path'],
                    project_name=project_name,
                    doc_type=pdf_info['type'],
                    doc_name=pdf_info.get('name')
                )
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {pdf_info['path']}: {e}")
                results.append({"document": pdf_info['path'], "error": str(e), "status": "failed"})
        
        return results
    
    def get_project_info(self, project_name):
        """Get project information"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Project {name: $project_name})-[:HAS_DOCUMENT]->(d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                RETURN p.name as project,
                       collect(DISTINCT d.name) as documents,
                       count(DISTINCT d) as doc_count,
                       count(c) as chunk_count
            """, project_name=project_name)
            
            record = result.single()
            if record:
                return dict(record)
            return None
    
    def close(self):
        """Close database connection"""
        self.driver.close()

# Usage Example
def main():
    # Initialize
    pdf_processor = SimplePDFToNeo4j(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j", 
        neo4j_password="your_password",
        openai_key="your_openai_key"
    )
    
    try:
        # Process single PDF
        pdf_processor.process_pdf(
            pdf_path="sample.pdf",
            project_name="My Project",
            doc_type="BRD",
            doc_name="Business Requirements"
        )
        
        # Process multiple PDFs
        pdf_list = [
            {"path": "brd.pdf", "type": "BRD", "name": "Business Requirements"},
            {"path": "prd.pdf", "type": "PRD", "name": "Product Requirements"},
            {"path": "qa.pdf", "type": "QA", "name": "QA Plan"}
        ]
        
        results = pdf_processor.process_multiple_pdfs("E-commerce Project", pdf_list)
        print("Processing Results:", results)
        
        # Get project info
        info = pdf_processor.get_project_info("E-commerce Project")
        print("Project Info:", info)
        
    finally:
        pdf_processor.close()

if __name__ == "__main__":
    main()
