import os
import json
from audio_p import transcribe_audio
from video_p import describe_video

# --- CONFIGURATION ---
AUDIO_FILE_PATH = 'input/audio.webm'
VIDEO_FILE_PATH = 'input/video.mp4'
OUTPUT_DIR = 'output'
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'processed_data.json')

def main():
    """
    Main pipeline to process podcast audio and video, then save the combined data.
    """
    print("--- Starting Podcast Processing Pipeline ---")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # --- Step 1: Process Audio ---
    print("\n[PHASE 1/2] Transcribing Audio...")
    audio_log = transcribe_audio(AUDIO_FILE_PATH)
    print("✅ Audio transcription complete.")
    
    # --- Step 2: Process Video ---
    print("\n[PHASE 2/2] Generating Video Descriptions...")
    # We can use a larger interval here for the initial high-level pass
    video_log = describe_video(VIDEO_FILE_PATH, interval_seconds=1)
    print("✅ Video description complete.")
    
    # --- Step 3: Combine and Save Data ---
    print("\nCombining and saving the final processed data...")
    
    combined_data = {
        "audio_log": audio_log,
        "video_log": video_log
    }
    
    with open(FINAL_OUTPUT_FILE, 'w') as f:
        json.dump(combined_data, f, indent=2)
        
    print(f"🎉 Success! Processed data saved to: {FINAL_OUTPUT_FILE}")
    print("--- Pipeline Finished ---")

if __name__ == '__main__':
    # Before running, make sure you have your Gemini API key set up for the next steps
    # os.environ['GOOGLE_API_KEY'] = 'YOUR_GEMINI_API_KEY'
    main()
