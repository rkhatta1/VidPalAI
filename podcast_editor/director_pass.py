import os
import json
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import google.generativeai as genai
from tqdm import tqdm
from rag_agent import get_index

# --- CONFIGURATION ---
STRUCTURAL_MAP_PATH = 'output/structural_map.json'
PROCESSED_DATA_PATH = 'output/processed_data_multi_cam.json'
OUTPUT_PATH = 'output/director_edits.json'
GEMINI_MODEL = "gemini-2.5-pro"

# --- Gemini API Configuration ---
try:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)


def format_multicam_local_context(start_time, end_time, master_audio_log, video_logs):
    """
    [MODIFIED] Gathers transcript from the master audio and VLM data from all cameras.
    """
    # Get transcript from the master audio log
    transcript = " ".join([
        item['word'] for item in master_audio_log
        if start_time <= item['start'] < end_time
    ])
    
    context_parts = [f"--- Master Transcript for this Segment ---\n{transcript}\n"]
    
    # Get visual cues from each camera's video log
    for cam_id, video_log in video_logs.items():
        visuals = [
            f"  - At {item['timestamp']:.1f}s: {item['description']}" for item in video_log 
            if start_time <= item['timestamp'] < end_time
        ]
        visual_summary = "\n".join(visuals)
        context_parts.append(f"--- Visual Cues from {cam_id} ---\n{visual_summary}\n")

    return "\n".join(context_parts)


def run_director_pass():
    """
    Iterates through chapters, using RAG and multi-camera local context to prompt Gemini
    for a detailed Edit Decision List.
    """
    print("--- Starting Director Pass (Pass 2) ---")

    # --- 1. Load All Necessary Data ---
    print("Loading structural map and multi-camera processed data...")
    with open(STRUCTURAL_MAP_PATH, 'r') as f:
        structural_map = json.load(f)
    with open(PROCESSED_DATA_PATH, 'r') as f:
        multi_cam_data = json.load(f)
        
    chapters = structural_map['chapters']
    master_audio_log = multi_cam_data.get('master_audio_log', [])
    video_logs = multi_cam_data.get('video_logs', {})

    # --- 2. Initialize RAG Agent ---
    index = get_index()
    if not index:
        print("Could not load the index. Aborting.")
        return
        
    query_engine = index.as_query_engine(similarity_top_k=2)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
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
        
        # [MODIFIED] Pass the new data structure to the context formatter
        local_context = format_multicam_local_context(start_time, end_time, master_audio_log, video_logs)

        # --- 3C. Craft the Director Prompt (No change needed here) ---
        system_prompt = """
        You are an expert multi-camera video director for a podcast with three cameras: 'cam_host', 'cam_guest', and 'cam_wide'. Your task is to create a precise Edit Decision List (EDL) for a segment of the podcast.
        You will be given global context from the entire conversation, and detailed local data (a master transcript and visual cues from all three cameras) for the specific segment you are editing.
        
        RULES:
        1. Prioritize the camera on the person who is currently speaking. Use the camera ID that best matches their role (e.g., 'cam_host' for the host).
        2. Cut to the other person for important non-verbal reactions (nodding, laughing, looking surprised).
        3. Use 'cam_wide' to establish the scene, during rapid back-and-forth dialogue, or when both speakers are interacting significantly.
        4. Do not make cuts too frequently. A shot should last at least 2-3 seconds.
        
        You MUST output a valid JSON object with a single key "cuts". The value should be a list of objects, where each object has:
        - "start_time": The start time of the shot in seconds.
        - "end_time": The end time of the shot in seconds.
        - "camera_id": A string, must be one of "cam_host", "cam_guest", or "cam_wide".
        """
        
        user_prompt = f"""
        **CHAPTER TO EDIT:** "{title}" (from {start_time}s to {end_time}s)

        **GLOBAL CONTEXT FROM RAG MEMORY:**
        {rag_context}

        **DETAILED MULTI-CAMERA LOCAL DATA FOR THIS CHAPTER:**
        {local_context}

        Based on all this information, generate the JSON for the Edit Decision List.
        """
        
        # --- 3D. Call the Gemini API (No change here) ---
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            chapter_edl = json.loads(response.text)
            final_edits.append({"chapter_title": title, "edl": chapter_edl})
        except Exception as e:
            print(f"\nError processing chapter '{title}': {e}")
            continue

    # --- 4. Save Final Output ---
    print("Saving the final director's edits...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump({"director_edits": final_edits}, f, indent=2)

    print(f"🎉 Success! Director's edits saved to: {OUTPUT_PATH}")
    print("--- Director Pass Finished ---")


if __name__ == '__main__':
    run_director_pass()
