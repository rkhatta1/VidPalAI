import whisper
import json
from tqdm import tqdm

SAMPLE_RATE = 16000 # Whisper's standard sample rate

def transcribe_audio(audio_path: str, duration_limit_seconds: int = None):
    """
    Transcribes a limited duration of an audio file using OpenAI's Whisper.
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
    
    print("Transcription complete. Formatting output...")
    
    word_log = []
    for segment in tqdm(result['segments'], desc="Processing audio segments"):
        for word_info in segment['words']:
            word_log.append({
                "word": word_info['word'].strip(),
                "start": word_info['start'],
                "end": word_info['end']
            })
            
    return word_log
