import os
import json
import ollama
from tqdm import tqdm

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = 'output/processed_data_multi_cam.json'
OUTPUT_PATH = 'output/structural_map.json'
OLLAMA_LLM = "gemma3:4b"

def run_producer_pass():
    """
    Analyzes the master transcript to create a high-level structural map.
    """
    print(f"--- Starting Producer Pass (Pass 1) ---")

    # --- 1. Prepare the Full Transcript ---
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    try:
        with open(PROCESSED_DATA_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Processed data file not found. Please run main.py first.")
        return

    # [MODIFIED] Use the master audio log directly
    audio_log = data.get('master_audio_log', [])
    if not audio_log:
        print(f"Error: Master audio log is empty in the processed data file.")
        return

    full_transcript = " ".join([item['word'] for item in audio_log])
    print("Full transcript prepared from master audio file.")

    # --- 2. Craft the Prompt for the LLM (No change here) ---
    system_prompt = """
    You are an expert podcast producer. Your task is to analyze the provided podcast transcript and divide it into logical chapters or segments based on topic changes, narrative shifts, or distinct parts of the conversation.

    For each chapter, you must provide:
    1. A concise `title`.
    2. A brief one-sentence `summary`.
    3. A `start_time` in seconds.
    4. An `end_time` in seconds.

    You MUST output your response as a single, valid JSON object. The object should have a single key named "chapters", which contains a list of the chapter objects.
    """

    # --- 3. Call the Ollama LLM (No change here) ---
    print(f"Sending request to local LLM '{OLLAMA_LLM}'. This may take several minutes...")
    try:
        client = ollama.Client()
        response = client.chat(
            model=OLLAMA_LLM,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': full_transcript}
            ],
            format='json'
        )
        print("LLM response received.")

        # --- 4. Parse and Save the Response (No change here) ---
        llm_output_str = response['message']['content']
        structured_result = json.loads(llm_output_str)

        with open(OUTPUT_PATH, 'w') as f:
            json.dump(structured_result, f, indent=2)
            
        print(f"🎉 Success! Structural map saved to: {OUTPUT_PATH}")

    except Exception as e:
        print(f"\nAn error occurred while communicating with the LLM: {e}")

    print("--- Producer Pass Finished ---")


if __name__ == '__main__':
    run_producer_pass()
