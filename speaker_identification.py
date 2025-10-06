import whisperx
import torch
import json
import os
from pydub import AudioSegment
import tempfile
from collections import defaultdict

# --- CONFIGURATION ---
MASTER_AUDIO_FILE = 'input/audio.mp3'
OUTPUT_DIR = 'output'
SPEAKER_MAP_FILE = os.path.join(OUTPUT_DIR, 'speaker_map.json')
PROCESS_DURATION_MINUTES = 5

# You'll need a HuggingFace token with access to pyannote models
HUGGINGFACE_TOKEN = os.getenv('HF_TOKEN', "default_token")

# Force CPU to avoid CUDA library issues (set to True if you want to try GPU)
USE_GPU = False

def prepare_audio_for_diarization(audio_path: str, target_duration_seconds: int = None):
    """
    Truncates audio to the target duration if specified.
    Returns path to processed audio and the actual duration used.
    """
    audio = AudioSegment.from_file(audio_path)
    audio_duration = len(audio) / 1000.0  # Convert to seconds
    
    print(f"Original audio duration: {audio_duration:.2f}s")
    
    # Determine target duration
    if target_duration_seconds:
        process_duration = min(target_duration_seconds, audio_duration)
    else:
        process_duration = audio_duration
    
    if process_duration < 10:
        raise ValueError(f"Audio too short: {audio_duration:.2f}s. Need at least 10 seconds.")
    
    # Truncate if needed
    if process_duration < audio_duration:
        print(f"Truncating to {process_duration}s")
        audio_segment = audio[:int(process_duration * 1000)]
        
        # Save to temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix='.wav',
            dir=OUTPUT_DIR
        )
        audio_segment.export(temp_file.name, format='wav')
        print(f"Created processed audio: {temp_file.name}")
        
        return temp_file.name, process_duration, True
    
    return audio_path, process_duration, False

def identify_speakers(audio_path: str, duration_limit_seconds: int = None):
    """
    Uses WhisperX to perform transcription and speaker diarization.
    Returns a list of speaker segments with timestamps.
    """
    # Use CPU to avoid CUDA issues
    device = "cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    if device == "cuda":
        print("✅ Using GPU for speaker diarization")
    else:
        print("⚠️ Using CPU for speaker diarization (this will be slower but more stable)")
    
    # Prepare audio
    processed_audio, actual_duration, is_temp = prepare_audio_for_diarization(
        audio_path, 
        duration_limit_seconds
    )
    
    try:
        # Step 1: Transcribe with Whisper
        print("Step 1/4: Loading Whisper model...")
        model = whisperx.load_model("base", device, compute_type=compute_type)
        
        print("Step 2/4: Transcribing audio...")
        audio = whisperx.load_audio(processed_audio)
        result = model.transcribe(audio, batch_size=8)  # Reduced batch size for CPU
        
        print(f"  Detected language: {result['language']}")
        
        # Step 2: Align whisper output
        print("Step 3/4: Aligning transcription...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
        
        # Step 3: Diarization
        print("Step 4/4: Identifying speakers...")
        
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HUGGINGFACE_TOKEN,
                device=device
            )
            
            # Run diarization
            diarize_segments = diarize_model(audio)
            
            # Assign speakers to words
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
        except Exception as e:
            print(f"  Warning: Diarization failed ({e}), using transcription only")
            diarize_segments = None
        
        # Extract speaker segments
        speaker_segments = []
        
        # Process segments with speaker info
        for segment in result["segments"]:
            speaker_id = segment.get("speaker", "SPEAKER_00")
            speaker_segments.append({
                "speaker_id": speaker_id,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        # If no speakers were assigned, extract from raw diarization
        if diarize_segments and all(seg["speaker_id"] == "SPEAKER_00" for seg in speaker_segments):
            print("  Using raw diarization segments...")
            speaker_segments = []
            for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
                speaker_segments.append({
                    "speaker_id": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "text": ""
                })
        
        unique_speakers = len(set(seg['speaker_id'] for seg in speaker_segments))
        print(f"✅ Identified {unique_speakers} unique speakers in {len(speaker_segments)} segments")
        print(f"   (processed {actual_duration:.0f} seconds / {actual_duration/60:.1f} minutes)")
        
        return speaker_segments
    
    finally:
        # Clean up temporary file
        if is_temp and os.path.exists(processed_audio):
            try:
                os.remove(processed_audio)
                print(f"Cleaned up temporary file: {processed_audio}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up {processed_audio}: {cleanup_error}")

def create_speaker_role_mapping(speaker_segments):
    """
    Maps speaker IDs to roles (host/guest).
    Uses heuristics: speaker with most talk time = host.
    """
    speaker_durations = defaultdict(float)
    for seg in speaker_segments:
        duration = seg['end'] - seg['start']
        speaker_durations[seg['speaker_id']] += duration
    
    # Sort by talk time (descending)
    sorted_speakers = sorted(
        speaker_durations.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Simple heuristic: most talkative = host, second = guest
    role_mapping = {}
    if len(sorted_speakers) >= 1:
        role_mapping[sorted_speakers[0][0]] = "host"
        print(f"  Host identified: {sorted_speakers[0][0]} ({sorted_speakers[0][1]:.1f}s total)")
    if len(sorted_speakers) >= 2:
        role_mapping[sorted_speakers[1][0]] = "guest"
        print(f"  Guest identified: {sorted_speakers[1][0]} ({sorted_speakers[1][1]:.1f}s total)")
    
    # Handle additional speakers
    for i, (speaker_id, duration) in enumerate(sorted_speakers[2:], start=3):
        role_mapping[speaker_id] = f"speaker_{i}"
        print(f"  Additional speaker: {speaker_id} ({duration:.1f}s total)")
    
    return role_mapping

def main():
    duration_seconds = PROCESS_DURATION_MINUTES * 60
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("--- Starting Speaker Identification with WhisperX ---\n")
    
    # Run diarization
    speaker_segments = identify_speakers(
        MASTER_AUDIO_FILE, 
        duration_limit_seconds=duration_seconds
    )
    
    print("\nCreating speaker role mapping...")
    # Create role mapping
    role_mapping = create_speaker_role_mapping(speaker_segments)
    
    # Save output
    output_data = {
        "speaker_segments": speaker_segments,
        "role_mapping": role_mapping
    }
    
    with open(SPEAKER_MAP_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Speaker map saved to: {SPEAKER_MAP_FILE}")

if __name__ == '__main__':
    main()
