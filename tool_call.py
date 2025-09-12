#!/usr/bin/env python3
"""
PoC: Agent + GraphCypherQAChain (langchain-neo4j)
- Uses Neo4jGraph (from langchain_neo4j) and GraphCypherQAChain.from_llm
- Wraps the chain as a Tool "CypherQA" so the agent can call it in natural language
- Instructs the agent to discover schema and emit test-cases as a JSON array
"""

import os
import re
import json
from dotenv import load_dotenv

# LangChain core
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType

# langchain-neo4j (recommended package)
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

load_dotenv()

# -----------------------
# CONFIG (ENV)
# -----------------------
NEO4J_URL = os.environ.get("NEO4J_URI")           # e.g. bolt://localhost:7687 or neo4j+s://...
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASS = os.environ.get("NEO4J_PASS")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

OUTFILE = os.environ.get("OUTPUT_FILE", "agent_generated_tests.json")

if not (NEO4J_URL and NEO4J_USER and NEO4J_PASS):
    raise RuntimeError("Please set NEO4J_URI, NEO4J_USER and NEO4J_PASS in environment or .env")

# -----------------------
# Create LLM (Azure-aware)
# -----------------------
llm_kwargs = {"temperature": 0.0, "max_tokens": 1500}
if AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT:
    llm = ChatOpenAI(openai_api_key=AZURE_API_KEY, deployment_name=AZURE_DEPLOYMENT, openai_api_base=AZURE_ENDPOINT, **llm_kwargs)
elif OPENAI_API_KEY:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, **llm_kwargs)
else:
    llm = ChatOpenAI(**llm_kwargs)

# -----------------------
# Neo4jGraph + CypherQAChain
# -----------------------
# Use the langchain-neo4j package's Neo4jGraph wrapper (recommended). :contentReference[oaicite:3]{index=3}
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASS)

# Create GraphCypherQAChain that will generate Cypher and run it on the graph.
# allow_dangerous_requests=False is safer; set True only if you accept generated Cypher risk.
# top_k controls how many rows from the generated Cypher are used as context for answer generation.
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=False,
    top_k=10,
    allow_dangerous_requests=False,  # use a read-only DB user in production
)

# -----------------------
# Tool wrapper for the agent
# -----------------------
def cypher_tool_fn(natural_language_query: str) -> str:
    """
    Wrap the GraphCypherQAChain: pass the natural-language query to the chain and return text.
    The chain itself will generate Cypher, run it against Neo4j, and return the text result.
    """
    try:
        res = cypher_chain.invoke({"query": natural_language_query})
        # GraphCypherQAChain returns a dict-like result; prefer "result" or convert to string
        if isinstance(res, dict):
            # 'result' key commonly contains the final answer text
            return str(res.get("result") or res)
        return str(res)
    except Exception as e:
        return f"[CypherQA ERROR] {e}"

cypher_tool = Tool(
    name="CypherQA",
    func=cypher_tool_fn,
    description=(
        "Run natural-language queries against the Neo4j graph. "
        "Examples: 'List top-level labels (views).', 'Show 10 sample records for label Policy', "
        "'Profile null counts and distinct values for field policyNumber on label Policy'."
    )
)

# -----------------------
# Build agent with tool
# -----------------------
tools = [cypher_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=8,
)

# -----------------------
# High-level instruction for the agent
# -----------------------
INSTRUCTION = """
You are a QA test-case generator. Use the CypherQA tool whenever you need to read the database.

Steps (follow exactly):
1) Discover node labels (top-level "views") in the graph using CypherQA.
2) For each view/label, use CypherQA to sample records and profile candidate fields (null counts, distinct counts, samples).
3) Auto-rank fields by importance (use distinct counts, non-null presence, sample content).
4) For each important field generate 1-4 presence-focused test cases. Each test case must be a JSON object with:
   - id (string)
   - label (string)
   - field (string)
   - priority (1-5)  # 1 highest
   - description (1-2 sentences)
   - check_query (either the natural-language query you would run with CypherQA, or Cypher)
   - hints (optional)
5) Produce a single JSON array (no extra commentary) containing all test-case objects. Keep JSON valid.

Start by listing labels and choose those that represent datasets/views. Then profile and generate tests.
"""

# -----------------------
# Run agent & save JSON
# -----------------------
if __name__ == "__main__":
    print("[INFO] Running agent (it will call the CypherQA tool several times)...")
    raw = agent.run(INSTRUCTION)

    # Extract first JSON array from the agent output (best-effort)
    m = re.search(r"(\[.*\])", raw, flags=re.S)
    json_text = m.group(1) if m else raw

    try:
        tests = json.loads(json_text)
    except Exception:
        # fallback: wrap the full output
        tests = [{"id": "agent_output_fallback", "raw": raw}]

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Written {len(tests)} testcases to {OUTFILE}")

New_instructions = """
Purpose
You will act as a Graph-RAG QA Test Generator. Your job is to explore a Neo4j graph (using the CypherQA tool for every database read), understand the schema and structure (top-level nodes / views, measure nodes, related sub-nodes), inspect representative samples of the data, rank important fields/measures, and produce human-readable test-cases that QA engineers can use to validate developers’ output.

Tool usage rules
• ALWAYS use the CypherQA tool to read the graph (labels, relationships, samples, counts, profiles).
• Never hardcode Cypher in your instruction text — use CypherQA in natural language; the chain will generate and execute Cypher.
• Only request read-only information (MATCH / RETURN / CALL db.*). If you detect any non-read Cypher, regenerate as read-only.
• Cache results you fetch (labels, samples, profiles) in your local context and reuse them — this is the RAG memory.

Step-by-step workflow (follow precisely)

Schema discovery
• Ask CypherQA for a list of all node labels and the count of nodes for each label.
• From this list identify candidate top-level labels (views) using heuristics:
– labels with largest node counts;
– labels that appear to be containers (e.g., have many outgoing :HAS_ROW or similar patterns);
– labels whose names look like source filenames or dataset names.
• Store the label list and counts in memory.

Structural inference (topology)
• For each candidate top-level label, ask CypherQA to list relationship types connecting it to other labels (incoming and outgoing) with counts.
• Build a simple hierarchy: view → direct record nodes → related subnodes (measures, attributes). Save that structure.

Sampling and profiling
• For each label you will process (or for a user-scoped label only if the user asked):
– Fetch up to N sample nodes (e.g., 20) as property maps.
– From samples extract property names and infer types (string / number / date / boolean / null).
– For each candidate property, ask CypherQA to compute basic stats: total count, null count, distinct count, and a few example values. Cache these profiles.

Auto-ranking of important fields/measures
• For each property/measure determine a numeric priority (1 highest … 5 lowest) using these heuristics:
– Primary-key candidate: high distinct count & very low nulls → high priority.
– Core measure: numeric field with many non-null values across samples → high priority.
– Business/categorical field: many distinct values or common non-null presence → medium/high priority.
– Low coverage or rare fields → lower priority.
• Record a one-line rationale for each priority (e.g., “distincts=500, null_ratio=0 → likely identifier”).

Types of test checks to generate (for each high-priority item produce 1–4 checks)
For each label and each high-priority property or relationship, create human-readable test cases of the following kinds where applicable:
• Presence: confirm the field exists and is populated for the expected fraction of records (often 100%).
• Non-null: check null counts and list sample node ids for missing values.
• Uniqueness/ID: verify distinct_count == total_count for identifier candidates.
• Type/format: verify type or regex (e.g., ISO date format, numeric).
• Range: numeric measures fall within observed sample min/max (flag extreme outliers).
• Referential integrity: for relationships, confirm referenced nodes exist (no orphan relations).
• Schema conformance: required properties appear on nodes of that label (no missing required fields).

Test-case content and formatting (human readable, not JSON)
For each test produce a concise block with these fields written in plain text:
• ID: a short unique id (Label_Field_001).
• Scope: global / view / label (name the view if scope=view).
• Target: property name or relationship type being tested.
• Condition: one of presence / non_null / uniqueness / type_format / range / referential_integrity / schema_conformance.
• Priority: 1–5.
• Description: 1–2 sentence user-story describing what to check and why.
• Check prompt for CypherQA: a natural-language prompt that, when passed to CypherQA, will return the metrics needed to decide pass/fail (examples below).
• Hints: expected thresholds or patterns (e.g., presence_ratio = 1.0, regex = ISO_DATE).
• Confidence rationale: short note linking the test to profile stats (e.g., “appears in 98% samples; distinct_count high → likely identifier”).

Output packaging
• Return the result as a human-readable document: a graph_summary (labels, counts, relations), per-label short profile table (field → total/nulls/distincts), and the list of test-case blocks as described above.
• If there are many tests, still produce a single readable file — group by label and paginate/chunk if needed, but keep the document complete.

User scope commands
• If the user asked to limit scope to a single view/label, restrict all reads and tests to that label and its directly related nodes.
• If user asks for “only measures”, restrict profiling and tests to nodes recognized as measures under each view.

Safety and validation
• Ensure every CypherQA prompt asks for read-only outputs. If CypherQA returns Cypher that looks non-read, regenerate the query.
• For each test provide a CypherQA prompt (not raw Cypher unless explicitly requested) so QA can re-run it through the CypherQA tool for verification.
"""
# #!/usr/bin/env python3
# """
# PoC: Agent + CypherQA tool
# - Registers a CypherQA tool (langchain-community)
# - Runs an agent that uses the tool (in natural language) to discover schema and generate tests
# - Saves final JSON testcases to a file
# """

# import os
# import json
# from dotenv import load_dotenv

# # LangChain core
# from langchain import LLMChain, PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.agents import Tool, initialize_agent, AgentType

# # langchain-community (Cypher QA)
# # Install: pip install langchain-community
# from langchain_community.graphs import Neo4jGraph
# from langchain_community.chains.graph_qa.cypher import CypherQAChain  # name may vary by version

# load_dotenv()

# # ---------------------
# # Config (env)
# # ---------------------
# NEO4J_URI = os.environ.get("NEO4J_URI")       # bolt://host:7687 or neo4j+s://...
# NEO4J_USER = os.environ.get("NEO4J_USER")
# NEO4J_PASS = os.environ.get("NEO4J_PASS")

# # OpenAI / Azure
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
# AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
# AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

# OUTFILE = os.environ.get("OUTPUT_FILE", "agent_generated_tests.json")

# if not (NEO4J_URI and NEO4J_USER and NEO4J_PASS):
#     raise RuntimeError("Please set NEO4J_URI, NEO4J_USER and NEO4J_PASS in environment or .env")

# # ---------------------
# # Create LLM (Azure-aware)
# # ---------------------
# llm_kwargs = {"temperature": 0.0, "max_tokens": 1500}
# if AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT:
#     # Many LangChain versions accept these azure args on ChatOpenAI; if not, use AzureChatOpenAI variant.
#     llm = ChatOpenAI(openai_api_key=AZURE_API_KEY, deployment_name=AZURE_DEPLOYMENT, openai_api_base=AZURE_ENDPOINT, **llm_kwargs)
# elif OPENAI_API_KEY:
#     llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, **llm_kwargs)
# else:
#     # fallback: rely on env-configured API key
#     llm = ChatOpenAI(**llm_kwargs)

# # ---------------------
# # Create Neo4j Graph object for CypherQAChain
# # ---------------------
# neo4j_graph = Neo4jGraph(uri=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)

# # Build the Cypher QA chain which converts NL -> Cypher -> executes -> returns results
# cypher_chain = CypherQAChain.from_llm(llm=llm, graph=neo4j_graph)

# # ---------------------
# # Wrap CypherQAChain as a LangChain Tool
# # ---------------------
# def cypher_qa_tool_fn(user_question: str) -> str:
#     """
#     Tool wrapper around CypherQAChain.run.
#     The agent will call this tool with natural-language prompts like:
#       - 'List top-level views (labels used as containers).'
#       - 'Show sample rows for label X.'
#       - 'Profile field Y in label X (total/nulls/distincts/samples).'
#     The chain will auto-generate Cypher, run it on Neo4j, and return results as text.
#     """
#     # cypher_chain.run expects an instruction string
#     try:
#         answer = cypher_chain.run(user_question)
#     except Exception as e:
#         # brief, safe error reporting back to the agent
#         answer = f"[CypherQA ERROR] {e}"
#     return str(answer)

# cypher_tool = Tool(
#     name="CypherQA",
#     func=cypher_qa_tool_fn,
#     description=(
#         "Use this tool to run natural-language queries against the Neo4j graph. "
#         "It will auto-generate Cypher and return results. Example calls:\n"
#         "- 'List all top-level labels used as Views'\n"
#         "- 'Return 10 sample records for label Policy'\n"
#         "- 'Show null counts and distinct counts for field policyNumber on label Policy'\n"
#         "Always return tabular or JSON-serializable answers."
#     )
# )

# # ---------------------
# # Build an Agent that has the Cypher tool available
# # ---------------------
# tools = [cypher_tool]

# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # simple agent type
#     verbose=True,
#     max_iterations=6  # small loop: agent can call tool multiple times; increase if needed
# )

# # ---------------------
# # High-level instruction for the agent
# # ---------------------
# INSTRUCTION = """
# You are a QA test-case generator. Your goal: discover the schema and generate presence-focused test cases
# for the data stored in the connected Neo4j graph.

# Rules:
# 1) Use the tool "CypherQA" whenever you need to read the database. Call it with natural-language prompts
#    like 'List labels', 'Sample records for label X', 'Profile field Y on label X'.
# 2) Do NOT assume any schema. Discover labels (views), record properties, and relationships using the tool.
# 3) For each label (or view), identify important fields automatically (use distinct counts, null counts, sample values).
# 4) For each important field generate 1-4 presence-focused test cases. Each test case must be a JSON object with:
#    - id (string)
#    - label (string)     # the node label / view
#    - field (string)
#    - priority (1-5)     # 1 = highest
#    - description (1-2 sentences)
#    - check_query (natural-language instruction that the CypherQA tool could run OR the Cypher the chain generates)
#    - hints (optional)
# 5) Produce a single JSON array (no extra commentary) containing all test-case objects. If it's large, produce as many as you can,
#    and keep output JSON-valid.
# 6) You may call CypherQA multiple times to profile and sample data. Make sure generated tests reference the label/field properly.

# Start by listing labels and pick the top-level ones (views). Then profile and generate tests.
# """

# # ---------------------
# # Run the agent and capture output
# # ---------------------
# if __name__ == "__main__":
#     print("[INFO] Running agent to auto-discover schema and generate tests. This may make several calls to CypherQA.")
#     result_text = agent.run(INSTRUCTION)

#     # try to parse JSON array from agent output; if agent already returns JSON, great.
#     # otherwise, we attempt a best-effort extraction of the first JSON array inside the text.
#     import re
#     m = re.search(r"(\[.*\])", result_text, flags=re.S)
#     json_text = m.group(1) if m else result_text

#     try:
#         tests = json.loads(json_text)
#     except Exception:
#         # fallback: wrap whole text as single string test (should not happen if agent followed rule)
#         tests = [{"id": "agent_output_fallback", "raw": result_text}]

#     # save
#     with open(OUTFILE, "w", encoding="utf-8") as f:
#         json.dump(tests, f, indent=2, ensure_ascii=False)

#     print(f"[DONE] Written {len(tests)} testcases to {OUTFILE}")
