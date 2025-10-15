import os
import glob
import pickle
from typing import List
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.pkl")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150))

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def load_documents_from_folder(folder: str) -> List[Document]:
    docs = []
    for p in glob.glob(os.path.join(folder, "*")):
        if p.lower().endswith(".pdf"):
            text = extract_text_from_pdf(p)
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(p)}))
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for d in tqdm(docs, desc="Chunking documents"):
        pieces = splitter.split_text(d.page_content)
        for i, p in enumerate(pieces):
            md = dict(d.metadata)
            md.update({"chunk": i})
            chunks.append(Document(page_content=p, metadata=md))
    return chunks

def build_faiss_index_from_folder(folder: str, save_path: str = INDEX_PATH):
    docs = load_documents_from_folder(folder)
    print(f"✅ Loaded {len(docs)} documents; chunking...")
    chunks = chunk_documents(docs)
    print(f"✅ Created {len(chunks)} chunks. Computing embeddings...")
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)
    with open(save_path, "wb") as f:
        pickle.dump(vectorstore, f)
    print(f"🎉 Saved FAISS index to {save_path}")
