import torch
from PIL import Image
import cv2
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_ID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200 # The special token ID the model looks for.

def describe_video(video_path: str, interval_seconds: int = 1):
    """
    Processes a video file using the official FastVLM-0.5B implementation,
    generating a description for a frame at each interval.
    
    Args:
        video_path: The path to the video file.
        interval_seconds: The interval in seconds to capture a frame.
        
    Returns:
        A list of dictionaries, where each dict contains a timestamp and a description.
    """
    print("Setting up VLM model with official implementation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # --- 1. Load Model and Tokenizer (as per official snippet) ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto", # device_map="auto" requires the 'accelerate' library
        trust_remote_code=True,
    )
    print("Model loaded successfully.")

    # --- 2. Prepare the prompt template ---
    # We build the prompt with a special <image> placeholder.
    messages = [{"role": "user", "content": "<image>\nDescribe what is happening in this scene."}]
    # This renders the chat messages into a single string for the model.
    rendered_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    
    # We split the prompt into the parts before and after the image placeholder.
    # This is an optimization so we don't have to re-tokenize the text for every frame.
    pre_prompt, post_prompt = rendered_prompt.split("<image>", 1)
    pre_ids = tokenizer(pre_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    img_token = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)

    # --- 3. Open and process the video file ---
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process_interval = int(fps * interval_seconds)
    
    video_log = []
    failed_frames_count = 0
    
    # Use tqdm for a nice progress bar
    for frame_num in tqdm(range(0, total_frames, frames_to_process_interval), desc="Processing video frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"\n Warning: Failed to read frame #{frame_num}. Skipping.")
            failed_frames_count += 1
            continue
            
        # --- 4. Process each frame ---
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Preprocess the image using the model's internal image processor
        pixel_values = model.get_vision_tower().image_processor(
            images=pil_image, return_tensors="pt"
        )["pixel_values"]
        pixel_values = pixel_values.to(model.device, dtype=model.dtype)
        
        # Splice the text and image tokens together to create the final input_ids
        input_ids = torch.cat([pre_ids, img_token, post_ids], dim=1).to(model.device)
        attention_mask = torch.ones_like(input_ids)

        # Generate the description
        with torch.no_grad():
            output = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                max_new_tokens=128,
            )
        
        # Decode and clean up the output text
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        # We split the response to remove the prompt part and get only the generated description
        try:
            description = full_response.split(post_prompt)[1].strip()
        except IndexError:
            description = full_response # Fallback if splitting fails

        timestamp = frame_num / fps
        video_log.append({
            "timestamp": timestamp,
            "description": description
        })
        
        torch.cuda.empty_cache()

    cap.release()
    print(f"\n--- Processing Summary ---")
    print(f"Successfully processed {len(video_log)} frames.")
    print(f"Failed to read {failed_frames_count} frames.")
    print("--------------------------")
    return video_log

# This allows you to test the script by running it directly
if __name__ == '__main__':
    import json
    video_file = 'input/podcast_video_h264_gpu.mp4'
    # Use a larger interval for faster testing
    descriptions = describe_video(video_file, interval_seconds=10) 
    
    # Save to a temporary file to check the output
    with open('output/vlm_test_new.json', 'w') as f:
        json.dump(descriptions, f, indent=2)
        
    print("Test VLM descriptions saved to output/vlm_test_new.json")
