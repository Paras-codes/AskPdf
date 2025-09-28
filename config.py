from dotenv import load_dotenv
import os

load_dotenv()

# LLM & Embeddings
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
HUGGINGFACE_TOKEN =  os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print(HUGGINGFACE_TOKEN)

# Directories
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Retrieval
TOP_K = int(os.getenv("TOP_K", 3))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0))
