import os
import json
import time
from audio_p import transcribe_audio
from video_p import describe_video
from speaker_identification import identify_speakers, create_speaker_role_mapping

# --- CONFIGURATION ---
MASTER_AUDIO_FILE = "input/audio.mp3"
VIDEO_FILES = {
  "cam_host": "input/cam_host.mp4",
  "cam_guest": "input/cam_guest.mp4",
  "cam_wide": "input/cam_wide.mp4",
}
OUTPUT_DIR = "output"
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_data_multi_cam.json")
SPEAKER_MAP_FILE = os.path.join(OUTPUT_DIR, "speaker_map.json")
PROCESS_DURATION_MINUTES = 5
VIDEO_INTERVAL_SECONDS = 2  # sample every N seconds per camera


def main():
  """
  Main pipeline (single-threaded).
  """
  duration_seconds = PROCESS_DURATION_MINUTES * 60
  print(
    f"--- Starting Multi-Camera Pipeline (First {PROCESS_DURATION_MINUTES} Minutes) ---"
  )

  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

  # --- PHASE 0: Speaker Identification ---
  print("\n[PHASE 0/3] Identifying Speakers")
  t0 = time.time()
  speaker_segments = identify_speakers(
    MASTER_AUDIO_FILE, duration_limit_seconds=duration_seconds
  )
  role_mapping = create_speaker_role_mapping(speaker_segments)

  speaker_data = {"speaker_segments": speaker_segments, "role_mapping": role_mapping}
  with open(SPEAKER_MAP_FILE, "w") as f:
    json.dump(speaker_data, f, indent=2)
  print(
    f"✅ Speaker identification complete. ({time.time() - t0:.1f}s) Saved: {SPEAKER_MAP_FILE}"
  )

  # --- PHASE 1: Transcribe Audio ---
  print("\n[PHASE 1/3] Transcribing Master Audio")
  t1 = time.time()
  master_audio_log = transcribe_audio(
    MASTER_AUDIO_FILE,
    duration_limit_seconds=duration_seconds,
    speaker_map_path=SPEAKER_MAP_FILE,
  )
  print(f"✅ Audio transcription complete. ({time.time() - t1:.1f}s)")

  # --- PHASE 2: Sequential Video Processing ---
  print("\n[PHASE 2/3] Processing cameras sequentially...")
  video_logs = {}

  for cam_id, file_path in VIDEO_FILES.items():
    if not os.path.exists(file_path):
      print(f"⚠️ Warning: {file_path} not found. Skipping {cam_id}.")
      continue

    print(f"[{cam_id}] Starting video processing...")
    t_cam = time.time()
    try:
      video_log = describe_video(
        file_path,
        interval_seconds=VIDEO_INTERVAL_SECONDS,
        duration_limit_seconds=duration_seconds,
        camera_id=cam_id,  # provides person context in prompts/log
      )
      video_logs[cam_id] = video_log
      print(
        f"[{cam_id}] ✅ Complete ({len(video_log)} samples, {time.time() - t_cam:.1f}s)"
      )
    except Exception as e:
      print(f"[{cam_id}] ❌ Error: {e}")

  print(f"✅ Processed {len(video_logs)} camera logs.")

  # --- PHASE 3: Save Combined Data ---
  print("\n[PHASE 3/3] Saving combined data...")
  combined_data = {
    "master_audio_log": master_audio_log,
    "video_logs": video_logs,
    "speaker_data": speaker_data,
  }

  with open(FINAL_OUTPUT_FILE, "w") as f:
    json.dump(combined_data, f, indent=2)

  print(f"🎉 Success! Data saved to: {FINAL_OUTPUT_FILE}")
  print("--- Pipeline Finished ---")


if __name__ == "__main__":
  main()
