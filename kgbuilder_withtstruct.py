import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

# PDF processing
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Neo4j
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# OpenAI and LangChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFToNeo4jGraphRAG:
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 openai_api_key: str = None):
        """
        Initialize the PDF to Neo4j Graph RAG system
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize connections
        self.driver = None
        self.embeddings = None
        self.llm = None
        self.text_splitter = None
        
        self._initialize_components()
        self._create_graph_schema()

    def _initialize_components(self):
        """Initialize Neo4j driver, embeddings, and other components"""
        try:
            # Neo4j driver
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
            
            # OpenAI components
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-ada-002"
            )
            
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-4",
                temperature=0
            )
            
            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _create_graph_schema(self):
        """Create the graph schema with constraints and indexes"""
        queries = [
            # Create constraints
            "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            
            # Create indexes
            "CREATE INDEX document_type IF NOT EXISTS FOR (d:Document) ON (d.type)",
            "CREATE INDEX chunk_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_index)",
            
            # Create vector index for chunks
            """
            CALL db.index.vector.createNodeIndex(
                'chunk_embeddings',
                'Chunk',
                'embedding',
                1536,
                'cosine'
            )
            """
        ]
        
        with self.driver.session() as session:
            for query in queries:
                try:
                    session.run(query)
                    logger.info(f"Executed: {query[:50]}...")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Schema element already exists, skipping...")
                    else:
                        logger.warning(f"Schema creation warning: {str(e)}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            raise

    def generate_document_id(self, project_name: str, document_name: str, document_type: str) -> str:
        """Generate unique document ID"""
        content = f"{project_name}_{document_name}_{document_type}"
        return hashlib.md5(content.encode()).hexdigest()

    def generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        content = f"{document_id}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()

    def process_pdf_to_graph(self, 
                           pdf_path: str, 
                           project_name: str, 
                           document_type: str,
                           document_name: str = None) -> Dict[str, Any]:
        """
        Process a PDF file and store it in Neo4j graph
        
        Args:
            pdf_path: Path to PDF file
            project_name: Name of the project
            document_type: Type of document (BRD, PRD, QA, etc.)
            document_name: Custom document name (optional)
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Extract document name if not provided
            if not document_name:
                document_name = Path(pdf_path).stem
            
            # Extract text from PDF
            logger.info(f"Processing PDF: {pdf_path}")
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                raise ValueError("No text extracted from PDF")
            
            # Generate document ID
            document_id = self.generate_document_id(project_name, document_name, document_type)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Generate embeddings for chunks
            chunk_embeddings = self.embeddings.embed_documents(chunks)
            
            # Store in Neo4j
            with self.driver.session() as session:
                # Create or merge project
                session.run("""
                    MERGE (p:Project {name: $project_name})
                    SET p.updated_at = datetime()
                """, project_name=project_name)
                
                # Create or merge document
                session.run("""
                    MATCH (p:Project {name: $project_name})
                    MERGE (d:Document {id: $document_id})
                    SET d.name = $document_name,
                        d.type = $document_type,
                        d.path = $pdf_path,
                        d.created_at = datetime(),
                        d.chunk_count = $chunk_count
                    MERGE (p)-[:HAS_DOCUMENT]->(d)
                """, 
                project_name=project_name,
                document_id=document_id,
                document_name=document_name,
                document_type=document_type,
                pdf_path=pdf_path,
                chunk_count=len(chunks))
                
                # Create chunks
                for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    chunk_id = self.generate_chunk_id(document_id, i)
                    
                    session.run("""
                        MATCH (d:Document {id: $document_id})
                        CREATE (c:Chunk {
                            id: $chunk_id,
                            text: $text,
                            chunk_index: $chunk_index,
                            embedding: $embedding,
                            created_at: datetime()
                        })
                        CREATE (d)-[:HAS_CHUNK]->(c)
                    """,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    chunk_index=i,
                    embedding=embedding)
            
            result = {
                "project_name": project_name,
                "document_id": document_id,
                "document_name": document_name,
                "document_type": document_type,
                "chunk_count": len(chunks),
                "status": "success"
            }
            
            logger.info(f"Successfully processed {pdf_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {str(e)}")
            raise

    def process_project_documents(self, 
                                project_name: str, 
                                documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents for a project
        
        Args:
            project_name: Name of the project
            documents: List of dictionaries with 'path', 'type', and optional 'name'
            
        Returns:
            List of processing results
        """
        results = []
        for doc in documents:
            try:
                result = self.process_pdf_to_graph(
                    pdf_path=doc['path'],
                    project_name=project_name,
                    document_type=doc['type'],
                    document_name=doc.get('name')
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "path": doc['path'],
                    "error": str(e),
                    "status": "failed"
                })
        return results

    def similarity_search(self, 
                         query: str, 
                         project_name: str = None,
                         document_type: str = None,
                         k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search on chunks
        
        Args:
            query: Search query
            project_name: Filter by project name
            document_type: Filter by document type
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build Cypher query
            cypher_query = """
                CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding) 
                YIELD node AS chunk, score
                MATCH (chunk)<-[:HAS_CHUNK]-(doc:Document)<-[:HAS_DOCUMENT]-(proj:Project)
            """
            
            params = {
                "k": k,
                "query_embedding": query_embedding
            }
            
            # Add filters
            conditions = []
            if project_name:
                conditions.append("proj.name = $project_name")
                params["project_name"] = project_name
            
            if document_type:
                conditions.append("doc.type = $document_type")
                params["document_type"] = document_type
            
            if conditions:
                cypher_query += " WHERE " + " AND ".join(conditions)
            
            cypher_query += """
                RETURN chunk.text AS text,
                       chunk.chunk_index AS chunk_index,
                       doc.name AS document_name,
                       doc.type AS document_type,
                       proj.name AS project_name,
                       score
                ORDER BY score DESC
            """
            
            with self.driver.session() as session:
                result = session.run(cypher_query, params)
                chunks = []
                for record in result:
                    chunks.append({
                        "text": record["text"],
                        "chunk_index": record["chunk_index"],
                        "document_name": record["document_name"],
                        "document_type": record["document_type"],
                        "project_name": record["project_name"],
                        "similarity_score": record["score"]
                    })
                
                return chunks
                
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise

    def create_rag_chain(self, project_name: str = None, document_type: str = None):
        """Create a RAG chain for question answering"""
        
        def retriever_func(query: str) -> List[Document]:
            chunks = self.similarity_search(
                query=query,
                project_name=project_name,
                document_type=document_type,
                k=5
            )
            
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        "document_name": chunk["document_name"],
                        "document_type": chunk["document_type"],
                        "project_name": chunk["project_name"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity_score": chunk["similarity_score"]
                    }
                )
                documents.append(doc)
            
            return documents
        
        # Custom prompt template
        prompt_template = """
        Use the following pieces of context from company documents to answer the question. 
        If you don't know the answer based on the context, just say that you don't know.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer: """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create a simple RAG function
        def rag_query(question: str) -> Dict[str, Any]:
            try:
                # Retrieve relevant documents
                docs = retriever_func(question)
                
                if not docs:
                    return {
                        "answer": "I don't have enough information to answer this question.",
                        "sources": []
                    }
                
                # Prepare context
                context = "\n\n".join([
                    f"Document: {doc.metadata['document_name']} ({doc.metadata['document_type']})\n{doc.page_content}"
                    for doc in docs
                ])
                
                # Generate answer
                formatted_prompt = prompt.format(context=context, question=question)
                response = self.llm.invoke(formatted_prompt)
                
                # Prepare sources
                sources = [
                    {
                        "document_name": doc.metadata["document_name"],
                        "document_type": doc.metadata["document_type"],
                        "project_name": doc.metadata["project_name"],
                        "chunk_index": doc.metadata["chunk_index"],
                        "similarity_score": doc.metadata["similarity_score"]
                    }
                    for doc in docs
                ]
                
                return {
                    "answer": response.content,
                    "sources": sources
                }
                
            except Exception as e:
                logger.error(f"RAG query failed: {str(e)}")
                return {
                    "answer": f"An error occurred: {str(e)}",
                    "sources": []
                }
        
        return rag_query

    def get_project_summary(self, project_name: str) -> Dict[str, Any]:
        """Get summary of a project's documents"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Project {name: $project_name})-[:HAS_DOCUMENT]->(d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                RETURN p.name AS project_name,
                       collect(DISTINCT {
                           name: d.name,
                           type: d.type,
                           chunk_count: d.chunk_count
                       }) AS documents,
                       count(DISTINCT d) AS document_count,
                       count(c) AS total_chunks
            """, project_name=project_name)
            
            record = result.single()
            if record:
                return dict(record)
            return None

    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

# Example usage and testing
def main():
    """Example usage of the PDFToNeo4jGraphRAG system"""
    
    # Initialize the system
    rag_system = PDFToNeo4jGraphRAG(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        openai_api_key="your_openai_api_key"
    )
    
    try:
        # Example: Process documents for a project
        project_name = "ECommerce Platform"
        documents = [
            {"path": "docs/ecommerce_brd.pdf", "type": "BRD", "name": "Business Requirements"},
            {"path": "docs/ecommerce_prd.pdf", "type": "PRD", "name": "Product Requirements"},
            {"path": "docs/ecommerce_qa.pdf", "type": "QA", "name": "Quality Assurance Plan"},
        ]
        
        # Process all documents
        results = rag_system.process_project_documents(project_name, documents)
        print("Processing Results:", results)
        
        # Get project summary
        summary = rag_system.get_project_summary(project_name)
        print("Project Summary:", summary)
        
        # Create RAG chain for the project
        rag_query = rag_system.create_rag_chain(project_name=project_name)
        
        # Example queries
        questions = [
            "What are the main functional requirements for the ecommerce platform?",
            "What testing strategies are mentioned in the QA documentation?",
            "What are the key features described in the PRD?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            result = rag_query(question)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['sources'])} documents")
            
    finally:
        rag_system.close()

if __name__ == "__main__":
    # Install required packages:
    # pip install neo4j langchain langchain-openai PyPDF2 python-dotenv openai
    main()
