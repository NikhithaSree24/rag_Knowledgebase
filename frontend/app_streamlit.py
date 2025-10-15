import streamlit as st
import requests
import tempfile
import os

st.title("RAG Knowledge Base Demo")

API_URL = st.text_input("API URL", "http://localhost:8000")

# 1️⃣ Upload & ingest files
uploaded = st.file_uploader("Upload PDF/TXT files", accept_multiple_files=True)
if st.button("Ingest files"):
    if uploaded:
        tmpdir = tempfile.mkdtemp()
        for f in uploaded:
            with open(os.path.join(tmpdir, f.name), "wb") as out:
                out.write(f.getbuffer())
        res = requests.post(f"{API_URL}/ingest", json={"folder": tmpdir})
        st.json(res.json())
    else:
        st.warning("Please upload at least one file.")

# 2️⃣ Query
query = st.text_input("Ask a question")
top_k = st.slider("Top-k documents to retrieve", min_value=1, max_value=10, value=4)
if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        res = requests.post(f"{API_URL}/query", json={"query": query, "top_k": top_k})
        st.json(res.json())
