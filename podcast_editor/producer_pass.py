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
    [MODIFIED] Gathers transcript with speaker labels and VLM data with person context.
    """
    # Format transcript with speaker labels
    transcript_parts = []
    current_speaker = None
    current_text = []
    
    for item in master_audio_log:
        if start_time <= item['start'] < end_time:
            word = item['word']
            speaker = item.get('speaker', 'unknown')
            
            if speaker != current_speaker:
                if current_text:
                    transcript_parts.append(f"[{current_speaker.upper()}]: {' '.join(current_text)}")
                    current_text = []
                current_speaker = speaker
            
            current_text.append(word)
    
    # Add final speaker segment
    if current_text:
        transcript_parts.append(f"[{current_speaker.upper()}]: {' '.join(current_text)}")
    
    transcript = "\n".join(transcript_parts)
    
    context_parts = [f"--- Master Transcript for this Segment ---\n{transcript}\n"]
    
    # Get visual cues with person labels from each camera's video log
    for cam_id, video_log in video_logs.items():
        visuals = []
        for item in video_log:
            if start_time <= item['timestamp'] < end_time:
                person = item.get('shows_person', 'unknown')
                desc = item['description']
                visuals.append(f"  - At {item['timestamp']:.1f}s (showing {person}): {desc}")
        
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
        
        local_context = format_multicam_local_context(start_time, end_time, master_audio_log, video_logs)

        # --- 3C. Craft the Director Prompt ---
        system_prompt = """
        You are an expert multi-camera video director for a podcast with three cameras: 'cam_host', 'cam_guest', and 'cam_wide'. Your task is to create a precise Edit Decision List (EDL) for a segment of the podcast.
        
        You will be given:
        1. Global context from the entire conversation
        2. Detailed local data with speaker-labeled transcript ([HOST] and [GUEST])
        3. Visual cues from all three cameras with person labels (showing host/guest/both)
        
        DIRECTOR'S RULES:
        1. **SPEAKER PRIORITY**: Cut to the camera showing the person who is currently speaking:
           - When [HOST] speaks → prefer 'cam_host'
           - When [GUEST] speaks → prefer 'cam_guest'
        
        2. **REACTION SHOTS**: Cut to the NON-speaking person's camera when they have important reactions:
           - Nodding, laughing, surprised expressions
           - Look for visual cues like "nodding" or "smiling" in the camera descriptions
        
        3. **WIDE SHOTS**: Use 'cam_wide' for:
           - Establishing shots at chapter beginnings
           - Rapid back-and-forth dialogue
           - Moments when both people are actively engaged
        
        4. **PACING**: 
           - Minimum shot duration: 2-3 seconds
           - Don't cut too frequently
           - Let emotional moments breathe
        
        5. **CONTEXT AWARENESS**: 
           - Use the speaker labels to know who's talking
           - Use the visual descriptions to catch non-verbal moments
           - Use the camera's "shows_person" info to ensure you're cutting to the right person
        
        You MUST output a valid JSON object with a single key "cuts". The value should be a list of objects, where each object has:
        - "start_time": The start time of the shot in seconds.
        - "end_time": The end time of the shot in seconds.
        - "camera_id": A string, must be one of "cam_host", "cam_guest", or "cam_wide".
        
        Example:
        {
          "cuts": [
            {"start_time": 0.0, "end_time": 3.5, "camera_id": "cam_wide"},
            {"start_time": 3.5, "end_time": 8.2, "camera_id": "cam_host"}
          ]
        }
        """
        
        user_prompt = f"""
        **CHAPTER TO EDIT:** "{title}" (from {start_time}s to {end_time}s)

        **GLOBAL CONTEXT FROM RAG MEMORY:**
        {rag_context}

        **DETAILED MULTI-CAMERA LOCAL DATA FOR THIS CHAPTER:**
        {local_context}

        Based on all this information, generate the JSON for the Edit Decision List.
        Remember: Match cameras to speakers, catch reactions, use wide shots strategically.
        """
        
        # --- 3D. Call the Gemini API ---
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
