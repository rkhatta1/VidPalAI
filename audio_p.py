import whisper
import json
from tqdm import tqdm

SAMPLE_RATE = 16000

def transcribe_audio(audio_path: str, duration_limit_seconds: int = None, speaker_map_path: str = None):
    """
    Transcribes audio and enriches with speaker labels if available.
    """
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    print(f"Loading audio from: {audio_path}")
    audio_array = whisper.load_audio(audio_path)
    
    if duration_limit_seconds:
        print(f"Processing only the first {duration_limit_seconds} seconds of audio.")
        num_samples = int(duration_limit_seconds * SAMPLE_RATE)
        audio_array = audio_array[:num_samples]

    print("Starting audio transcription...")
    result = model.transcribe(audio_array, word_timestamps=True)
    
    # Load speaker map if available
    speaker_data = None
    if speaker_map_path:
        try:
            with open(speaker_map_path, 'r') as f:
                speaker_data = json.load(f)
            print("✅ Speaker map loaded. Enriching transcript...")
        except FileNotFoundError:
            print("⚠️ Speaker map not found. Continuing without speaker labels.")
    
    print("Transcription complete. Formatting output...")
    
    word_log = []
    for segment in tqdm(result['segments'], desc="Processing audio segments"):
        for word_info in segment['words']:
            word_entry = {
                "word": word_info['word'].strip(),
                "start": word_info['start'],
                "end": word_info['end']
            }
            
            # Add speaker label if available
            if speaker_data:
                speaker_label = get_speaker_at_time(
                    word_info['start'], 
                    speaker_data['speaker_segments'],
                    speaker_data['role_mapping']
                )
                word_entry["speaker"] = speaker_label
            
            word_log.append(word_entry)
            
    return word_log

def get_speaker_at_time(timestamp, speaker_segments, role_mapping):
    """
    Finds which speaker is active at a given timestamp.
    """
    for seg in speaker_segments:
        if seg['start'] <= timestamp <= seg['end']:
            speaker_id = seg['speaker_id']
            return role_mapping.get(speaker_id, speaker_id)
    return "unknown"
