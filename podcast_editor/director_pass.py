import os
import json
# import ollama # Commented out the ollama import
import google.generativeai as genai
from tqdm import tqdm
from agent_setup import get_index
from dotenv import load_dotenv

# --- CONFIGURATION ---
STRUCTURAL_MAP_PATH = 'output/structural_map.json'
PROCESSED_DATA_PATH = 'output/processed_data_10min.json'
OUTPUT_PATH = 'output/director_edits.json'
# OLLAMA_LLM = "gemma3:4b" # Using the efficient Gemma 2 model
GEMINI_MODEL = "gemini-2.5-pro"

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)


def format_local_context(start_time, end_time, audio_log, video_log):
    """Gathers and formats the granular data for a specific time segment."""
    # Get all words spoken in the segment
    transcript = " ".join([
        item['word'] for item in audio_log
        if start_time <= item['start'] < end_time
    ])

    # Get all VLM descriptions for the segment
    visuals = [
        f"At {item['timestamp']:.1f}s: {item['description']}" for item in video_log
        if start_time <= item['timestamp'] < end_time
    ]
    visual_summary = "\n".join(visuals)

    return f"TRANSCRIPT FOR THIS SEGMENT:\n{transcript}\n\nVISUAL CUES FOR THIS SEGMENT:\n{visual_summary}"


def run_director_pass():
    """
    Iterates through chapters, uses RAG to get context, and prompts an LLM
    to create a detailed Edit Decision List for each chapter.
    """
    print("--- Starting Director Pass (Pass 2) ---")

    # --- 1. Load All Necessary Data ---
    print("Loading structural map and processed data...")
    with open(STRUCTURAL_MAP_PATH, 'r') as f:
        structural_map = json.load(f)
    with open(PROCESSED_DATA_PATH, 'r') as f:
        processed_data = json.load(f)

    audio_log = processed_data['audio_log']
    video_log = processed_data['video_log']
    chapters = structural_map['chapters']

    # --- 2. Initialize the RAG Agent (Now a single function call!) ---
    index = get_index()
    if not index:
        print("Could not load the index. Aborting.")
        return

    query_engine = index.as_query_engine(similarity_top_k=2)
    
    # [NEW] Initialize the Gemini Model
    model = genai.GenerativeModel(GEMINI_MODEL)
    # llm_client = ollama.Client() # Commented out Ollama client

    final_edits = []

    # --- 3. Main Processing Loop ---
    for chapter in tqdm(chapters, desc="Directing Chapters"):
        title = chapter['title']
        summary = chapter['summary']
        start_time = chapter['start_time']
        end_time = chapter['end_time']
        rag_query = f"What is the most important context from the rest of the podcast related to this topic: {summary}"
        rag_response = query_engine.query(rag_query)
        rag_context = rag_response.response
        local_context = format_local_context(start_time, end_time, audio_log, video_log)

        # --- 3C. Craft the Director Prompt (Now with a clear system prompt) ---
        system_prompt = """
        You are an expert multi-camera video director for a podcast. Your task is to create a precise Edit Decision List (EDL) for a segment of the podcast.
        You will be given global context from the entire conversation, and detailed local data (transcript and visual cues) for the specific segment you are editing.

        RULES:
        1. Prioritize the camera on the person who is currently speaking.
        2. Cut to the other person for important non-verbal reactions (nodding, laughing, looking surprised).
        3. Use a 'wide_shot' to establish the scene or during rapid back-and-forth dialogue.
        4. Do not make cuts too frequently. A shot should last at least 2-3 seconds.

        You MUST output a valid JSON object with a single key "cuts". The value should be a list of objects, where each object has:
        - "start_time": The start time of the shot in seconds.
        - "end_time": The end time of the shot in seconds.
        - "camera_id": A string, either "cam_host", "cam_guest", or "cam_wide".
        """

        user_prompt = f"""
        **CHAPTER TO EDIT:** "{title}" (from {start_time}s to {end_time}s)

        **GLOBAL CONTEXT FROM RAG MEMORY:**
        {rag_context}

        **DETAILED LOCAL DATA FOR THIS CHAPTER:**
        {local_context}

        Based on all this information, generate the JSON for the Edit Decision List.
        """

        # --- 3D. Call the LLM to get the edit ---
        try:
            # [NEW] Call the Gemini API
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            chapter_edl = json.loads(response.text)
            
            # --- Commented out Ollama call ---
            # response = llm_client.chat(
            #     model=OLLAMA_LLM,
            #     messages=[
            #         {'role': 'system', 'content': system_prompt},
            #         {'role': 'user', 'content': user_prompt}
            #     ],
            #     format='json'
            # )
            # chapter_edl = json.loads(response['message']['content'])
            
            final_edits.append({"chapter_title": title, "edl": chapter_edl})
        except Exception as e:
            print(f"\nError processing chapter '{title}': {e}")
            continue

    # --- 4. Save Final Output ---
    print("Saving the final director's edits...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(final_edits, f, indent=2)

    print(f"🎉 Success! Director's edits saved to: {OUTPUT_PATH}")
    print("--- Director Pass Finished ---")


if __name__ == '__main__':
    run_director_pass()
