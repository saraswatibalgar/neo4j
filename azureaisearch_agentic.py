import os
from langchain.chat_models import init_chat_model
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import Tool
from langchain_community.retrievers import AzureAISearchRetriever

# --- Azure environment setup ---
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_SEARCH_INDEX = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]

# --- LLM (Azure OpenAI) ---
llm = init_chat_model(
    f"azure_openai:{AZURE_OPENAI_DEPLOYMENT}",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_api_key=AZURE_OPENAI_KEY,
    temperature=0
)

# --- Azure AI Search retriever ---
retriever = AzureAISearchRetriever(
    content_key="content",
    top_k=5,
    index_name=AZURE_SEARCH_INDEX
)

# --- Define retriever as a LangChain Tool ---
def search_azure(query: str) -> str:
    """Retrieve relevant info from Azure AI Search"""
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

retriever_tool = Tool(
    name="AzureSearch",
    func=search_azure,
    description="Use this tool to search Azure AI Search for relevant information"
)

# --- Initialize the Agent ---
agent = initialize_agent(
    tools=[retriever_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Use the Agent ---
if __name__ == "__main__":
    instruction = "Generate 5 JSON formatted functional test cases for login API using related context."
    result = agent.invoke(instruction)
    print("\n--- Final Output ---\n")
    print(result["output"])
