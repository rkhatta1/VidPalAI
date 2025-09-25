import os
import json
from mem0 import Mem0
from tqdm import tqdm
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = 'output/processed_data.json'
# We'll create a unique collection for each podcast edit
COLLECTION_NAME = 'podcast-session-1' 
# The duration of each memory chunk in seconds. 5-10 seconds is a good starting point.
CHUNK_DURATION_SECONDS = 5.0 
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def populate_memory_from_log():
    """
    Loads the processed audio/video data, chunks it into coherent intervals,
    and populates a mem0 collection with it.
    """
    print(f"Loading processed data from: {PROCESSED_DATA_PATH}")
    try:
        with open(PROCESSED_DATA_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{PROCESSED_DATA_PATH}'.")
        print("Please run main.py first to generate the data.")
        return

    audio_log = data.get('audio_log', [])
    video_log = data.get('video_log', [])

    if not audio_log:
        print("Warning: Audio log is empty. Cannot populate memory.")
        return

    # --- Initialize mem0 ---
    # Each podcast can be its own collection.
    print(f"Initializing local embedding model: {LOCAL_EMBED_MODEL}")
    # This line tells the underlying framework to use our chosen local model
    # for all subsequent embedding tasks.
    Settings.embed_model = HuggingFaceEmbedding(model_name=LOCAL_EMBED_MODEL)

    # Now, when we initialize Mem0, it will automatically use the model we just set.
    print(f"Initializing mem0 collection '{COLLECTION_NAME}'...")
    mem0 = Mem0(collection_name=COLLECTION_NAME)
    
    # Determine the total duration of the podcast from the last word
    total_duration = audio_log[-1]['end']
    
    print(f"Total podcast duration: {total_duration:.2f} seconds.")
    print(f"Processing and adding data to mem0 in {CHUNK_DURATION_SECONDS}s chunks...")
    
    # --- The "Bucketing" Logic ---
    # We iterate through time, creating a chunk for each interval.
    num_chunks = 0
    for start_time in tqdm(range(0, int(total_duration), int(CHUNK_DURATION_SECONDS)), desc="Creating memory chunks"):
        end_time = start_time + CHUNK_DURATION_SECONDS
        
        # 1. Gather all words spoken in this time chunk
        words_in_chunk = [
            item['word'] for item in audio_log 
            if start_time <= item['start'] < end_time
        ]
        transcript_chunk = " ".join(words_in_chunk)
        
        # 2. Find the most relevant video description for this chunk
        # We find the latest description that occurred at or before the chunk started.
        relevant_descriptions = [
            item['description'] for item in video_log
            if item['timestamp'] <= start_time
        ]
        visual_context = relevant_descriptions[-1] if relevant_descriptions else "No visual data."
        
        # 3. Combine into a single, coherent memory string
        if transcript_chunk:
            memory_text = f"[VISUAL: {visual_context}] TRANSCRIPT: {transcript_chunk}"
            
            # 4. Add the chunk to mem0 with the timestamp as metadata
            mem0.add(
                memory_text,
                metadata={'timestamp': start_time}
            )
            num_chunks += 1

    print(f"\n🎉 Success! Populated mem0 collection '{COLLECTION_NAME}' with {num_chunks} memory chunks.")

if __name__ == '__main__':
    populate_memory_from_log()
