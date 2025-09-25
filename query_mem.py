import os
import json
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION (MUST MATCH populate_memory.py) ---
SESSION_ID = 'podcast-session-1' 
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384 

def query_memories():
    """
    Initializes mem0 with the correct configuration and performs a search
    on a specified session's memory.
    """
    # 1. Define the exact same configuration as the population script.
    # This is crucial for accessing the correct database and using the correct
    # embedding model to encode your query.
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": EMBEDDING_DIM
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": LOCAL_EMBED_MODEL
            }
        }
    }
    
    print("Initializing mem0...")
    mem0 = Memory.from_config(config)
    
    # 2. Define your search query.
    # This should be a question about something you expect was in the first 10 mins.
    query = "What did the guest say about their trip?"
    
    print(f"\nSearching memories for session '{SESSION_ID}' with query: \"{query}\"")
    
    # 3. Perform the search.
    # You must provide the same `user_id` used when adding the memories.
    results = mem0.search(query=query, user_id=SESSION_ID)
    print("\n--- DEBUG INFO ---")
    print(f"Type of 'results': {type(results)}")
    print(f"Raw 'results' value: {repr(results)}")
    print("--- END DEBUG INFO ---\n")
    # --- END DEBUGGING BLOCK ---

    if not results or results == ["results"]: # Also check for the erroneous value
        print("\nNo relevant memories found or received invalid response.")
        return
    # 4. Process and display the results.
    if not results:
        print("\nNo relevant memories found.")
        return
        
    print(f"\nFound {len(results)} relevant memories:")
    for i, result_str in enumerate(results):
        try:
            # 1. Parse the JSON string into a Python dictionary
            result_dict = json.loads(result_str)

            # 2. Now, safely access the keys from the dictionary
            score = result_dict.get('score', 0.0)
            memory_text = result_dict.get('text', 'N/A')
            metadata = result_dict.get('metadata', {})
            timestamp = metadata.get('timestamp', 'N/A') # Metadata is already a dict here

            print("\n" + "-"*10)
            print(f"Result {i+1} | Relevance Score: {score:.2f}")
            print(f"Timestamp: ~{timestamp}s")
            print(f"Memory: {memory_text}")
            print("-"*10)

        except (json.JSONDecodeError, TypeError) as e:
            print(f"\nCould not parse result #{i+1}: {result_str}")
            print(f"Error: {e}")

if __name__ == '__main__':
    query_memories()
