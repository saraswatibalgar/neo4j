import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import Tool

# --- Azure Setup ---
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_SEARCH_SERVICE = os.environ["AZURE_AI_SEARCH_SERVICE_NAME"]
AZURE_SEARCH_API_KEY = os.environ["AZURE_AI_SEARCH_API_KEY"]
AZURE_SEARCH_INDEX = "langchain-vector-demo"  # change if needed

# --- 1. Initialize Embeddings ---
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",  # or your embedding model name
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)

# --- 2. Create AzureSearch Vector Store ---
vector_store = AzureSearch(
    embedding_function=embeddings.embed_query,
    azure_search_endpoint=AZURE_SEARCH_SERVICE,
    azure_search_key=AZURE_SEARCH_API_KEY,
    index_name=AZURE_SEARCH_INDEX,
)

# --- 3. Create Retriever from Vector Store ---
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# --- 4. Define Retriever Tool ---
def azure_retrieve_tool(query: str) -> str:
    """Retrieve relevant info from Azure AI Search Vector Index"""
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

retriever_tool = Tool(
    name="AzureVectorSearch",
    func=azure_retrieve_tool,
    description="Use this tool to search Azure AI Search vector index for relevant context."
)

# --- 5. Initialize Azure OpenAI LLM ---
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    temperature=0
)

# --- 6. Initialize Agent ---
agent = initialize_agent(
    tools=[retriever_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- 7. Run single instruction ---
if __name__ == "__main__":
    instruction = "Generate 5 functional test cases for user registration API using relevant documentation."
    result = agent.invoke(instruction)
    print("\n--- Final Output ---\n")
    print(result["output"])
