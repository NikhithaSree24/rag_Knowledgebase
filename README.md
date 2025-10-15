# Knowledge-base Search Engine (RAG)

This is a **Retrieval-Augmented Generation (RAG) Knowledge-base Search Engine**.  
It allows you to search across multiple documents (PDF or text) and get **concise, synthesized answers** using a **local language model**.



## Features

- Upload multiple documents and ingest them into a local FAISS index
- Split documents into chunks and generate embeddings locally
- Query documents and get synthesized answers from a local LLM (`flan-t5-base`)
- Fully offline: no OpenAI API key, no billing, no quota issues
- Frontend interface built with Streamlit



## Folder Structure

rag_project/
├── api/
│ ├── main.py
│ ├── rag_utils.py
│ └── ingest.py
├── frontend/
│ └── app_streamlit.py
├── docs/ (sample PDFs or text)
├── .env
├── requirements.txt
├── .gitignore
└── README.md



## Setup Instructions

1. **Clone the repository**


git clone <your-repo-url>
cd rag_project
Create virtual environment and install dependencies


python -m venv venv
venv\Scripts\activate     # Windows
 or
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
Run the backend (FastAPI)


uvicorn api.main:app --reload
Run the frontend (Streamlit)



streamlit run frontend/app_streamlit.py
Upload documents and start asking questions!

Technical Details
Embeddings: all-MiniLM-L6-v2 (via Sentence Transformers)

Vector Store: FAISS (local)

LLM: google/flan-t5-base (via Hugging Face Transformers)

Backend: FastAPI

Frontend: Streamlit
