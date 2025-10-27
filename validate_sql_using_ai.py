from databricks import sql
from openai import AzureOpenAI
import json

# ---------- CONFIG ----------
AZURE_OPENAI_ENDPOINT = "https://<your-endpoint>.openai.azure.com/"
AZURE_OPENAI_KEY = "<your-key>"
AZURE_OPENAI_MODEL = "gpt-4o-mini"

DATABRICKS_SERVER_HOSTNAME = "<your-databricks-sql-endpoint>"
DATABRICKS_HTTP_PATH = "<your-http-path>"
DATABRICKS_ACCESS_TOKEN = "<your-personal-access-token>"

# ---------- INIT ----------
client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT)

# ---------- STEP 1: FETCH SCHEMA ----------
def get_schema_text():
    schema_text = ""
    with sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_ACCESS_TOKEN,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [row[1] for row in cursor.fetchall()]  # row = (db, tableName, isTemp)
            for t in tables[:5]:  # limit for simplicity
                cursor.execute(f"DESCRIBE TABLE {t}")
                cols = [f"{r[0]} ({r[1]})" for r in cursor.fetchall()]
                schema_text += f"\nTable {t}: " + ", ".join(cols)
    return schema_text.strip()

# ---------- STEP 2: SYNTAX CHECK ----------
def check_syntax(query):
    try:
        with sql.connect(
            server_hostname=DATABRICKS_SERVER_HOSTNAME,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_ACCESS_TOKEN,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"EXPLAIN {query}")
        return 100
    except Exception as e:
        print("Syntax Error:", e)
        return 0

# ---------- STEP 3: LLM SEMANTIC VALIDATION ----------
def llm_semantic_check(query, intent, schema_text):
    prompt = f"""
Schema:
{schema_text}

Intent:
{intent}

SQL Query:
{query}

You are an SQL Validator Agent.
Rate how well this query matches the intent and schema.
Return JSON only:
{{"score": 0-100, "explanation": "text", "fix_suggestion": "text"}}
"""
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"score": 0, "explanation": "Invalid LLM output", "fix_suggestion": ""}

# ---------- STEP 4: MAIN VALIDATION ----------
def validate_query(query, intent):
    schema_text = get_schema_text()
    syntax_score = check_syntax(query)
    if syntax_score == 0:
        return {"final_score": 0, "reason": "SQL syntax error"}

    semantic = llm_semantic_check(query, intent, schema_text)
    final_score = (syntax_score * 0.4) + (semantic["score"] * 0.6)
    return {
        "final_score": final_score,
        "syntax_score": syntax_score,
        "semantic_score": semantic["score"],
        "explanation": semantic["explanation"],
        "fix_suggestion": semantic["fix_suggestion"]
    }

# ---------- STEP 5: RUN ----------
if __name__ == "__main__":
    intent = "Get total sales per region for last month"
    sql_query = "SELECT region, SUM(sales) FROM orders WHERE month='2025-09' GROUP BY region"

    result = validate_query(sql_query, intent)
    print(result)

    if result["final_score"] < 80:
        print("❌ Query invalid")
    else:
        print("✅ Query valid")
