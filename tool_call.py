#!/usr/bin/env python3
"""
PoC: Agent + CypherQA tool
- Registers a CypherQA tool (langchain-community)
- Runs an agent that uses the tool (in natural language) to discover schema and generate tests
- Saves final JSON testcases to a file
"""

import os
import json
from dotenv import load_dotenv

# LangChain core
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType

# langchain-community (Cypher QA)
# Install: pip install langchain-community
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import CypherQAChain  # name may vary by version

load_dotenv()

# ---------------------
# Config (env)
# ---------------------
NEO4J_URI = os.environ.get("NEO4J_URI")       # bolt://host:7687 or neo4j+s://...
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASS = os.environ.get("NEO4J_PASS")

# OpenAI / Azure
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

OUTFILE = os.environ.get("OUTPUT_FILE", "agent_generated_tests.json")

if not (NEO4J_URI and NEO4J_USER and NEO4J_PASS):
    raise RuntimeError("Please set NEO4J_URI, NEO4J_USER and NEO4J_PASS in environment or .env")

# ---------------------
# Create LLM (Azure-aware)
# ---------------------
llm_kwargs = {"temperature": 0.0, "max_tokens": 1500}
if AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT:
    # Many LangChain versions accept these azure args on ChatOpenAI; if not, use AzureChatOpenAI variant.
    llm = ChatOpenAI(openai_api_key=AZURE_API_KEY, deployment_name=AZURE_DEPLOYMENT, openai_api_base=AZURE_ENDPOINT, **llm_kwargs)
elif OPENAI_API_KEY:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, **llm_kwargs)
else:
    # fallback: rely on env-configured API key
    llm = ChatOpenAI(**llm_kwargs)

# ---------------------
# Create Neo4j Graph object for CypherQAChain
# ---------------------
neo4j_graph = Neo4jGraph(uri=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)

# Build the Cypher QA chain which converts NL -> Cypher -> executes -> returns results
cypher_chain = CypherQAChain.from_llm(llm=llm, graph=neo4j_graph)

# ---------------------
# Wrap CypherQAChain as a LangChain Tool
# ---------------------
def cypher_qa_tool_fn(user_question: str) -> str:
    """
    Tool wrapper around CypherQAChain.run.
    The agent will call this tool with natural-language prompts like:
      - 'List top-level views (labels used as containers).'
      - 'Show sample rows for label X.'
      - 'Profile field Y in label X (total/nulls/distincts/samples).'
    The chain will auto-generate Cypher, run it on Neo4j, and return results as text.
    """
    # cypher_chain.run expects an instruction string
    try:
        answer = cypher_chain.run(user_question)
    except Exception as e:
        # brief, safe error reporting back to the agent
        answer = f"[CypherQA ERROR] {e}"
    return str(answer)

cypher_tool = Tool(
    name="CypherQA",
    func=cypher_qa_tool_fn,
    description=(
        "Use this tool to run natural-language queries against the Neo4j graph. "
        "It will auto-generate Cypher and return results. Example calls:\n"
        "- 'List all top-level labels used as Views'\n"
        "- 'Return 10 sample records for label Policy'\n"
        "- 'Show null counts and distinct counts for field policyNumber on label Policy'\n"
        "Always return tabular or JSON-serializable answers."
    )
)

# ---------------------
# Build an Agent that has the Cypher tool available
# ---------------------
tools = [cypher_tool]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # simple agent type
    verbose=True,
    max_iterations=6  # small loop: agent can call tool multiple times; increase if needed
)

# ---------------------
# High-level instruction for the agent
# ---------------------
INSTRUCTION = """
You are a QA test-case generator. Your goal: discover the schema and generate presence-focused test cases
for the data stored in the connected Neo4j graph.

Rules:
1) Use the tool "CypherQA" whenever you need to read the database. Call it with natural-language prompts
   like 'List labels', 'Sample records for label X', 'Profile field Y on label X'.
2) Do NOT assume any schema. Discover labels (views), record properties, and relationships using the tool.
3) For each label (or view), identify important fields automatically (use distinct counts, null counts, sample values).
4) For each important field generate 1-4 presence-focused test cases. Each test case must be a JSON object with:
   - id (string)
   - label (string)     # the node label / view
   - field (string)
   - priority (1-5)     # 1 = highest
   - description (1-2 sentences)
   - check_query (natural-language instruction that the CypherQA tool could run OR the Cypher the chain generates)
   - hints (optional)
5) Produce a single JSON array (no extra commentary) containing all test-case objects. If it's large, produce as many as you can,
   and keep output JSON-valid.
6) You may call CypherQA multiple times to profile and sample data. Make sure generated tests reference the label/field properly.

Start by listing labels and pick the top-level ones (views). Then profile and generate tests.
"""

# ---------------------
# Run the agent and capture output
# ---------------------
if __name__ == "__main__":
    print("[INFO] Running agent to auto-discover schema and generate tests. This may make several calls to CypherQA.")
    result_text = agent.run(INSTRUCTION)

    # try to parse JSON array from agent output; if agent already returns JSON, great.
    # otherwise, we attempt a best-effort extraction of the first JSON array inside the text.
    import re
    m = re.search(r"(\[.*\])", result_text, flags=re.S)
    json_text = m.group(1) if m else result_text

    try:
        tests = json.loads(json_text)
    except Exception:
        # fallback: wrap whole text as single string test (should not happen if agent followed rule)
        tests = [{"id": "agent_output_fallback", "raw": result_text}]

    # save
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Written {len(tests)} testcases to {OUTFILE}")
