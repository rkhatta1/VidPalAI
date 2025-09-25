import torch
from PIL import Image
import cv2
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time

MODEL_ID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200

def describe_video(video_path: str, interval_seconds: int = 1, duration_limit_seconds: int = None):
    """
    Processes a limited duration of a video file using a single thread.
    """
    print("Setting up VLM model with official implementation...")
    print("Loading model in 8-bit mode to conserve VRAM...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda",
        trust_remote_code=True,
    )
    print("Model loaded successfully.")

    # Prepare the prompt template
    messages = [{"role": "user", "content": "<image>\nDescribe what is happening in this scene."}]
    rendered_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    pre_prompt, post_prompt = rendered_prompt.split("<image>", 1)
    pre_ids = tokenizer(pre_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    img_token = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)

    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error: Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # NEW: Limit the number of frames to process
    max_frames_to_process = total_frames
    if duration_limit_seconds:
        print(f"Processing only the first {duration_limit_seconds} seconds of video.")
        max_frames_to_process = int(fps * duration_limit_seconds)
    
    frames_to_process_indices = range(0, min(total_frames, max_frames_to_process), int(fps * interval_seconds))
    
    video_log = []
    
    for frame_num in tqdm(frames_to_process_indices, desc="Processing video frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"\nWarning: Failed to read frame #{frame_num}. Skipping.")
            continue
            
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        pixel_values = model.get_vision_tower().image_processor(
            images=pil_image, return_tensors="pt"
        )["pixel_values"].to(model.device, model.dtype)
        
        input_ids = torch.cat([pre_ids, img_token, post_ids], dim=1).to(model.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                max_new_tokens=128,
            )
        
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            description = full_response.split(post_prompt)[1].strip()
        except IndexError:
            description = full_response

        timestamp = frame_num / fps
        video_log.append({
            "timestamp": timestamp,
            "description": description
        })
        torch.cuda.empty_cache()

    cap.release()
    return video_log
