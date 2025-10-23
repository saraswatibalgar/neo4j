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
import json

# --- Build full‐row context from manual (base) dataset ---
base_context = ""
for _, mrow in manual_df.iterrows():
    row_dict = mrow.to_dict()
    base_context += json.dumps(row_dict, ensure_ascii=False) + "\n---\n"

# --- Prepare LLM prompt templates ---
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_msg = SystemMessagePromptTemplate.from_template(
    "You are a QA test-case comparison expert. You have access to a list of manual test cases (full details). You will now be given a generated test case (full details). "
    "Your job: decide whether the generated test case is semantically equivalent (in purpose, context, outcome) to any of the manual ones. "
    "If yes, identify the matching manual TestcaseID and provide a reason. If no, say it is unmatched."
)

human_msg = HumanMessagePromptTemplate.from_template(
    "Manual test cases (for context):\n{base_context}\n\n"
    "Generated test case to evaluate (full details):\n{cand_full}\n\n"
    "Question: Does the generated test case match one of the manual ones? "
    "If yes respond exactly as JSON: {{\"same\": true, \"matched_manual_id\": \"<manualID>\", \"reason\": \"<short reason>\"}}. "
    "If no respond exactly as JSON: {{\"same\": false, \"reason\": \"<short reason>\"}}"
)

chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

# --- Instantiate your Azure LLM ---
from langchain_openai.chat_models.azure import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="YOUR_AZURE_DEPLOYMENT_NAME",
    api_version="2024-05-01-preview",
    temperature=0.0
)

# --- Comparison function using full row input ---
def compare_generated_to_manual(cand_row_dict: dict) -> dict:
    cand_full_json = json.dumps(cand_row_dict, ensure_ascii=False)
    prompt_messages = chat_prompt.format_prompt(
        base_context=base_context,
        cand_full=cand_full_json
    ).to_messages()
    
    resp = llm.invoke(prompt_messages)
    text = resp.generations[0][0].text.strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"same": False,
                  "reason": f"Could not parse response: {text}"}
    return result

# --- Build report structure ---
report = {
    "in_both": [],
    "generated_not_in_manual": [],
    "manual_missed_by_generated": []
}

matched_manual_ids = set()

# --- Compare each generated test case against the manual set ---
for _, grow in generated_df.iterrows():
    cand_dict = grow.to_dict()
    result = compare_generated_to_manual(cand_dict)
    
    if result.get("same"):
        report["in_both"].append({
            "manual_id": result.get("matched_manual_id"),
            "generated_row": cand_dict,
            "reason": result.get("reason")
        })
        matched_manual_ids.add(result.get("matched_manual_id"))
    else:
        report["generated_not_in_manual"].append({
            "generated_row": cand_dict,
            "reason": result.get("reason")
        })

# --- Identify manual test cases that were not matched by any generated case ---
for _, mrow in manual_df.iterrows():
    manual_id = mrow["TestcaseID"]  # adjust column name if different
    if manual_id not in matched_manual_ids:
        report["manual_missed_by_generated"].append({
            "manual_id": manual_id,
            "manual_row": mrow.to_dict()
        })

# --- Output the report ---
print(json.dumps(report, indent=2, ensure_ascii=False))
with open("comparison_report_fullrow.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
