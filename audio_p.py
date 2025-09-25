import whisper
import json
from tqdm import tqdm

def transcribe_audio(audio_path: str):
    """
    Transcribes an audio file using OpenAI's Whisper and returns
    a list of words with their timestamps.
    
    Args:
        audio_path: The path to the audio file.
        
    Returns:
        A list of dictionaries, where each dict contains a word and its start/end times.
    """
    print("Loading Whisper model...")
    # Using the 'base' model for a good balance of speed and accuracy.
    # For higher accuracy, you can use 'medium' or 'large'.
    model = whisper.load_model("base")
    
    print(f"Starting audio transcription for: {audio_path}")
    # The word_timestamps=True option is crucial for our granular log
    result = model.transcribe(audio_path, word_timestamps=True)
    
    print("Transcription complete. Formatting output...")
    
    word_log = []
    # The result contains segments, and each segment has words
    for segment in tqdm(result['segments'], desc="Processing segments"):
        for word_info in segment['words']:
            word_log.append({
                "word": word_info['word'].strip(),
                "start": word_info['start'],
                "end": word_info['end']
            })
            
    return word_log

# This allows you to test the script by running it directly
if __name__ == '__main__':
    audio_file = 'input/audio.webm'
    transcript = transcribe_audio(audio_file)
    
    # Save to a temporary file to check the output
    with open('output/transcript_test.json', 'w') as f:
        json.dump(transcript, f, indent=2)
        
    print("Test transcription saved to output/transcript_test.json")
