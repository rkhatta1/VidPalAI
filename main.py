import os
import json
from audio_p import transcribe_audio
from video_p import describe_video

# --- CONFIGURATION ---
AUDIO_FILE_PATH = 'input/audio.webm'
VIDEO_FILE_PATH = 'input/podcast_video_h264_gpu.mp4' 
OUTPUT_DIR = 'output'
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'processed_data_10min.json')
# Set the duration limit in minutes for the test run
PROCESS_DURATION_MINUTES = 10

def main():
    """
    Main pipeline to process a limited duration of the podcast.
    """
    duration_seconds = PROCESS_DURATION_MINUTES * 60
    print(f"--- Starting Podcast Processing Pipeline (First {PROCESS_DURATION_MINUTES} Minutes) ---")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"\n[PHASE 1/2] Transcribing Audio for {PROCESS_DURATION_MINUTES} minutes...")
    audio_log = transcribe_audio(AUDIO_FILE_PATH, duration_limit_seconds=duration_seconds)
    print("✅ Audio transcription complete.")
    
    print(f"\n[PHASE 2/2] Generating Video Descriptions for {PROCESS_DURATION_MINUTES} minutes...")
    video_log = describe_video(
        VIDEO_FILE_PATH, 
        interval_seconds=1, 
        duration_limit_seconds=duration_seconds
    )
    print("✅ Video description complete.")
    
    print("\nCombining and saving the final processed data...")
    combined_data = {
        "audio_log": audio_log,
        "video_log": video_log
    }
    
    with open(FINAL_OUTPUT_FILE, 'w') as f:
        json.dump(combined_data, f, indent=2)
        
    print(f"🎉 Success! Processed data for the first {PROCESS_DURATION_MINUTES} minutes saved to: {FINAL_OUTPUT_FILE}")
    print("--- Pipeline Finished ---")

if __name__ == '__main__':
    main()
