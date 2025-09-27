from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

import os
from config import PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def ingest_pdfs(file_paths: list[str], persist_dir=PERSIST_DIR):
    embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    total_chunks = 0
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["doc_id"] = f"{filename}_chunk_{i}"

        vectordb.add_documents(chunks)
        total_chunks += len(chunks)

    vectordb.persist()
    return { "total_chunks": total_chunks}
