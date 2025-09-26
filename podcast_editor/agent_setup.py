import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

# --- Centralized Configuration ---
PERSIST_DIR = "./llama_index_storage"
COLLECTION_NAME = "podcast_session_1"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_LLM = "gemma3:4b"

def get_index():
    """
    The single source of truth for initializing and loading our LlamaIndex agent.
    It configures the local embedder, the Ollama LLM, and the persistent storage.
    """
    # --- Configure Global Settings ---
    # This ensures any part of LlamaIndex uses our chosen models.
    print(f"Configuring embed model '{EMBED_MODEL}' and LLM '{OLLAMA_LLM}'...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=OLLAMA_LLM, request_timeout=120.0)

    # --- Load the index from the persistent ChromaDB store ---
    print(f"Loading existing index from {PERSIST_DIR}...")
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # This call will now use the global Settings we defined above
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    print("Index loaded successfully.")
    return index
