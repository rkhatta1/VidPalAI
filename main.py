import os
import json
from audio_p import transcribe_audio
from video_p import describe_video

# --- CONFIGURATION ---
# [MODIFIED] Define the single master audio file and the three video files
MASTER_AUDIO_FILE = 'input/audio.mp3'
VIDEO_FILES = {
    "cam_host": "input/cam_host.mp4",
    "cam_guest": "input/cam_guest.mp4",
    "cam_wide": "input/cam_wide.mp4"
}
OUTPUT_DIR = 'output'
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'processed_data_multi_cam.json')
PROCESS_DURATION_MINUTES = 10

def main():
    """
    Main pipeline for a multi-camera setup with a single master audio file.
    """
    duration_seconds = PROCESS_DURATION_MINUTES * 60
    print(f"--- Starting Multi-Camera Pipeline (First {PROCESS_DURATION_MINUTES} Minutes) ---")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # --- 1. Transcribe the Master Audio File (ONCE) ---
    print(f"\n[PHASE 1/2] Transcribing Master Audio File: {MASTER_AUDIO_FILE}")
    master_audio_log = transcribe_audio(MASTER_AUDIO_FILE, duration_limit_seconds=duration_seconds)
    print("✅ Master audio transcription complete.")
    
    # --- 2. Generate Video Descriptions for Each Camera Angle ---
    video_logs = {}
    for cam_id, file_path in VIDEO_FILES.items():
        print(f"\n[PHASE 2/2] Generating Video Descriptions for: {cam_id}")
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {cam_id} at {file_path}. Skipping.")
            continue
        
        video_log = describe_video(
            file_path, 
            interval_seconds=1, 
            duration_limit_seconds=duration_seconds
        )
        video_logs[cam_id] = video_log
        print(f"✅ Video description for {cam_id} complete.")

    # --- 3. Combine and Save the Final Data ---
    print("\nCombining and saving the final multi-camera processed data...")
    combined_data = {
        "master_audio_log": master_audio_log,
        "video_logs": video_logs
    }
    
    with open(FINAL_OUTPUT_FILE, 'w') as f:
        json.dump(combined_data, f, indent=2)
        
    print(f"🎉 Success! Processed multi-camera data saved to: {FINAL_OUTPUT_FILE}")
    print("--- Pipeline Finished ---")

if __name__ == '__main__':
    main()
