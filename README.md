# Knowledge-base Search Engine (RAG)

This is a **Retrieval-Augmented Generation (RAG) Knowledge-base Search Engine**.  
It allows you to search across multiple documents (PDF or text) and get **concise, synthesized answers** using a **local language model**.



## Features

- Upload multiple documents and ingest them into a local FAISS index
- Split documents into chunks and generate embeddings locally
- Query documents and get synthesized answers from a local LLM (`flan-t5-base`)
- Fully offline: no OpenAI API key, no billing, no quota issues
- Frontend interface built with Streamlit


## Setup Instructions

1.**Clone the repository**


git clone <your-repo-url>
cd rag_project
Create virtual environment and install dependencies

2.**Create virtual environment and install dependencies**


python -m venv venv
venv\Scripts\activate     # Windows
OR
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
Run the backend (FastAPI)

3.**Run the backend (FastAPI)**


uvicorn api.main:app --reload

4.**Run the frontend (Streamlit)**


streamlit run frontend/app_streamlit.py

5. **Upload documents and start asking questions!**

## Technical Details
Embeddings: all-MiniLM-L6-v2 (via Hugging Face / Sentence Transformers)

Vector Store: FAISS (local)

LLM: google/flan-t5-base (local, via Hugging Face Transformers)

Backend: FastAPI

Frontend: Streamlit

