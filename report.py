import os
import pandas as pd
import json

from langchain_openai.chat_models.azure import AzureChatOpenAI  # latest import style :contentReference[oaicite:2]{index=2}
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# --- Setup Azure LLM ---
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "<YOUR_ENDPOINT>")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "<YOUR_KEY>")
# (Also set OPENAI_API_VERSION etc if needed) :contentReference[oaicite:3]{index=3}

llm = AzureChatOpenAI(
    azure_deployment="YOUR_DEPLOYMENT_NAME",
    api_version="2024-05-01-preview",
    temperature=0.0,
)

# --- Load CSVs ---
manual_df = pd.read_csv("actual_testcases.csv")
generated_df = pd.read_csv("generated_testcases.csv")

# --- Choose roles ---
base_df = manual_df       # load all manual cases in memory/context
loop_df = generated_df    # we will check generated vs manual

# --- Build base context string ---
# Assume columns: TestcaseID, Description, ExpectedResult
base_context = ""
for _, row in base_df.iterrows():
    base_context += f"ID: {row['TestcaseID']}\nDescription: {row['Description']}\nExpected: {row['ExpectedResult']}\n---\n"

# --- Prepare prompt templates ---
system_msg = SystemMessagePromptTemplate.from_template(
    "You are a QA engineer. You know manual test cases and you will judge whether a generated test case matches one of the manual ones by purpose and outcome."
)

human_msg = HumanMessagePromptTemplate.from_template(
    "Here are the manual test cases (for your context):\n{base_context}\n\n"
    "Now consider this generated test case:\n"
    "ID: {cand_id}\nDescription: {cand_desc}\nExpected: {cand_expected}\n\n"
    "Question: Does this generated test case match any manual test case (yes/no)?\n"
    "If yes, respond as JSON: {{\"same\": true, \"matched_manual_id\": \"<ID>\", \"reason\": \"...\"}}\n"
    "If no, respond as JSON: {{\"same\": false, \"reason\": \"...\"}}"
)

prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

# --- Comparison function ---
def compare_case(cand_id, cand_desc, cand_expected):
    formatted = prompt.format_prompt(
        base_context=base_context,
        cand_id=cand_id,
        cand_desc=cand_desc,
        cand_expected=cand_expected,
    )
    resp = llm.invoke(formatted.to_messages())
    text = resp.generations[0][0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"same": False, "reason": f"Could not parse response: {text}"}

# --- Build report ---
report = {
    "in_both": [],
    "generated_not_in_manual": [],
    "manual_missed_by_generated": []
}

matched_manual_ids = set()

# Loop over generated cases
for _, row in loop_df.iterrows():
    cand_id = row["TestcaseID"]
    cand_desc = row["Description"]
    cand_expected = row.get("ExpectedResult", "")
    result = compare_case(cand_id, cand_desc, cand_expected)
    if result.get("same"):
        report["in_both"].append({
            "manual_id": result.get("matched_manual_id"),
            "generated_id": cand_id,
            "reason": result.get("reason")
        })
        matched_manual_ids.add(result.get("matched_manual_id"))
    else:
        report["generated_not_in_manual"].append({
            "generated_id": cand_id,
            "description": cand_desc,
            "reason": result.get("reason")
        })

# Now check manual cases that were never matched
for _, row in base_df.iterrows():
    mid = row["TestcaseID"]
    if mid not in matched_manual_ids:
        report["manual_missed_by_generated"].append({
            "manual_id": mid,
            "description": row["Description"]
        })

# Save or print report
print(json.dumps(report, indent=2))
with open("comparison_report.json", "w") as f:
    json.dump(report, f, indent=2)
