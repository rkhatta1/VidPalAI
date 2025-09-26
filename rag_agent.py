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
PROCESSED_DATA_PATH = 'output/processed_data_10min.json'
PERSIST_DIR = "./llama_index_storage" # Directory to save the persistent index
COLLECTION_NAME = "podcast_session_1"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_LLM = "deepseek-r1:1.5b"
CHUNK_DURATION_SECONDS = 10.0 # Using slightly larger chunks for better context

def load_and_prepare_documents():
    """
    Loads our custom JSON log, processes it into coherent chunks,
    and returns a list of LlamaIndex Document objects.
    """
    print(f"Loading and preparing documents from {PROCESSED_DATA_PATH}...")
    try:
        with open(PROCESSED_DATA_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Processed data file not found. Please run main.py first.")
        return []

    audio_log = data.get('audio_log', [])
    video_log = data.get('video_log', [])
    if not audio_log: return []

    documents = []
    total_duration = audio_log[-1]['end']

    # --- The "Bucketing" Logic ---
    for start_time in range(0, int(total_duration), int(CHUNK_DURATION_SECONDS)):
        end_time = start_time + CHUNK_DURATION_SECONDS
        
        words_in_chunk = [item['word'] for item in audio_log if start_time <= item['start'] < end_time]
        transcript_chunk = " ".join(words_in_chunk)
        
        relevant_descriptions = [item['description'] for item in video_log if item['timestamp'] <= start_time]
        visual_context = relevant_descriptions[-1] if relevant_descriptions else "No visual data."
        
        if transcript_chunk:
            memory_text = f"[Time: ~{start_time}s]\n[VISUAL: {visual_context}]\n[TRANSCRIPT: {transcript_chunk}]"
            
            # Create a LlamaIndex Document for each chunk
            doc = Document(
                text=memory_text,
                metadata={"start_time": start_time, "end_time": end_time}
            )
            documents.append(doc)
            
    print(f"Created {len(documents)} document chunks.")
    return documents

def get_index():
    """
    Initializes the embedding model, LLM, and vector store.
    Builds a new index if one doesn't exist, otherwise loads the existing one.
    """
    # --- 1. Configure Global Settings ---
    print(f"Configuring embed model '{EMBED_MODEL}' and LLM '{OLLAMA_LLM}'...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=OLLAMA_LLM, request_timeout=120.0)

    # --- 2. Check if index exists and load or build it ---
    if not os.path.exists(PERSIST_DIR):
        print("No existing index found. Building a new one...")
        documents = load_and_prepare_documents()
        if not documents:
            print("No documents to index. Exiting.")
            return None

        # Create a persistent ChromaDB vector store
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print(f"Index built and persisted to {PERSIST_DIR}")
    else:
        print(f"Loading existing index from {PERSIST_DIR}...")
        # Load the existing index
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        print("Index loaded successfully.")
        
    return index

if __name__ == '__main__':
    # Make sure Ollama is running before starting the script!
    index = get_index()
    
    if index:
        # --- 3. Query the index ---
        query_engine = index.as_query_engine()
        query = "What parellels did they draw with YouTube creators and politics?"
        
        print(f"\n--- Querying Agent ---")
        print(f"Query: {query}")
        response = query_engine.query(query)
        
        print("\nResponse:")
        print(response.response)
        
        print("\nSource Nodes (The context retrieved from memory):")
        for node in response.source_nodes:
            print("-" * 20)
            print(f"Similarity Score: {node.score:.4f}")
            print(f"Node Text:\n{node.text}")
            print("-" * 20)
