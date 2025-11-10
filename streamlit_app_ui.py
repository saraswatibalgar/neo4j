# app.py
import streamlit as st
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import IndexDocumentsBatch
import openai

# azure config
SEARCH_ENDPOINT = "https://xxxxx.search.windows.net"
SEARCH_KEY = "xxxx"
SEARCH_INDEX = "docs"
search_client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, AzureKeyCredential(SEARCH_KEY))

openai.api_key = "xxxx"

# simple system prompt
SYS_PROMPT = "You are expert PM. Create user stories from below knowledge."

st.sidebar.title("Menu")
page = st.sidebar.radio("", ["Upload Docs", "Generate Stories"])


if page == "Upload Docs":
    st.title("Upload PDF / Word")

    files = st.file_uploader("upload", accept_multiple_files=True)

    if st.button("Push to Azure Search"):
        batch = IndexDocumentsBatch()
        for f in files:
            text = f.read().decode(errors='ignore')
            batch.add_upload({"id": f.name, "content": text})
        search_client.index_documents(batch)
        st.success("uploaded")


if page == "Generate Stories":
    st.title("Generate User Stories")

    query = st.text_input("prompt")

    if st.button("Generate"):
        # get all docs from index
        docs = search_client.search("*")
        full = ""
        for d in docs:
            full += d["content"] + "\n"

        prompt = SYS_PROMPT + "\n\n" + full + "\n\nUser Query:" + query

        out = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        res = out.choices[0].message["content"]
        st.write(res)
