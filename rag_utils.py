# api/rag_utils.py
import os
import pickle
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Use your existing FAISS index path
INDEX_PATH = "faiss_index.pkl"

# Load FAISS vectorstore
def load_vectorstore(path=INDEX_PATH) -> FAISS:
    with open(path, "rb") as f:
        vs = pickle.load(f)
    return vs

# Prompt template
RAG_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        "Use the context to answer the user's question concisely.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
)

# Load local LLM model
TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-base")
MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Generate text using Hugging Face model
def generate_answer(prompt: str, max_length: int = 512):
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True)
    outputs = MODEL.generate(**inputs, max_length=max_length)
    answer = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return answer

# Retrieve documents
def retrieve_docs(query: str, k: int = 4):
    vs = load_vectorstore()
    emb = OpenAIEmbeddings(model_name="all-MiniLM-L6-v2")  # local embeddings
    results = vs.similarity_search(query, k=k)
    return results

# Main RAG function
def synthesize_answer(query: str, top_k: int = 4) -> dict:
    docs = retrieve_docs(query, k=top_k)
    context_texts = []
    sources = []
    for d in docs:
        context_texts.append(d.page_content)
        src = d.metadata.get("source", "")
        chunk = d.metadata.get("chunk", "")
        sources.append(f"{src}#{chunk}")
    context_combined = "\n\n---\n\n".join(context_texts)
    prompt = RAG_PROMPT.format(query=query, context=context_combined[:30000])
    answer = generate_answer(prompt)
    return {"answer": answer, "sources": sources, "retrieved_count": len(docs)}
