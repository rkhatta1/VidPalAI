import os
import json
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = 'output/processed_data_multi_cam.json'
PERSIST_DIR = "./llama_index_storage" # Directory to save the persistent index
COLLECTION_NAME = "podcast_session_multicam"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_LLM = "deepseek-r1:1.5b"
CHUNK_DURATION_SECONDS = 10.0 # Using slightly larger chunks for better context

def load_and_prepare_documents():
    """
    [MODIFIED] Loads the multi-camera JSON, processes it into coherent chunks,
    and returns a list of LlamaIndex Document objects.
    """
    print(f"Loading and preparing documents from {PROCESSED_DATA_PATH}...")
    try:
        with open(PROCESSED_DATA_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Processed data file not found. Please run main.py first.")
        return []

    audio_log = data.get('master_audio_log', [])
    video_logs = data.get('video_logs', {})
    if not audio_log: return []

    documents = []
    total_duration = audio_log[-1]['end']

    # --- The "Bucketing" Logic ---
    for start_time in range(0, int(total_duration), int(CHUNK_DURATION_SECONDS)):
        end_time = start_time + CHUNK_DURATION_SECONDS
        
        # Get transcript chunk from the master audio log
        words_in_chunk = [item['word'] for item in audio_log if start_time <= item['start'] < end_time]
        transcript_chunk = " ".join(words_in_chunk)
        
        # [NEW] Combine visual context from all cameras for this chunk
        visual_context = ""
        for cam_id, video_log in video_logs.items():
            # Find the most recent description from this camera for the chunk's start time
            relevant_descriptions = [item['description'] for item in video_log if item['timestamp'] <= start_time]
            if relevant_descriptions:
                last_description = relevant_descriptions[-1]
                visual_context += f"[{cam_id} VISUAL]: {last_description}\n"
        
        if transcript_chunk:
            memory_text = f"[Time: ~{start_time}s]\n{visual_context}[TRANSCRIPT: {transcript_chunk}]"
            
            doc = Document(
                text=memory_text,
                metadata={"start_time": start_time, "end_time": end_time}
            )
            documents.append(doc)
            
    print(f"Created {len(documents)} document chunks from multi-camera data.")
    return documents

def get_index():
    """
    Initializes models and vector store. Builds a new index if one doesn't exist,
    otherwise loads the existing one. (Logic inside is mostly unchanged).
    """
    print(f"Configuring embed model '{EMBED_MODEL}' and LLM '{OLLAMA_LLM}'...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=OLLAMA_LLM, request_timeout=120.0)

    # Note: If you re-run this, you might want to delete the old ./llama_index_storage directory
    # to ensure the index is rebuilt with the new multi-camera data structure.
    if not os.path.exists(PERSIST_DIR):
        print("No existing index found. Building a new one...")
        documents = load_and_prepare_documents()
        if not documents:
            print("No documents to index. Exiting.")
            return None

        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print(f"Index built and persisted to {PERSIST_DIR}")
    else:
        print(f"Loading existing index from {PERSIST_DIR}...")
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        print("Index loaded successfully.")
        
    return index

if __name__ == '__main__':
    # This script's main purpose is to build or load the index.
    # The query part is for testing.
    index = get_index()
    if index:
        print("\n✅ RAG Agent Index is ready.")
