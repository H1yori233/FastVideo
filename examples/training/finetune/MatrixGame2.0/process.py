import json
import os
import sys
import numpy as np
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
BASE_OUTPUT_DIR = 'car_8'
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
NUM_WORKERS = 8  # Number of parallel workers
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'videos')
VALIDATE_IMG_DIR = os.path.join(BASE_OUTPUT_DIR, 'validate')
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(VALIDATE_IMG_DIR, exist_ok=True)

# Action Mapping Configuration
KEY_TO_INDEX = {
    'W': 0,
    'D': 1,
    'A': 2,
    'S': 3
}

CAM_VALUE = 0.1
VIEW_ACTION_TO_MOUSE = {
    "up": [CAM_VALUE, 0.0],
    "down": [-CAM_VALUE, 0.0],
    "left": [0.0, -CAM_VALUE],
    "right": [0.0, CAM_VALUE],
    "up_right": [CAM_VALUE, CAM_VALUE],
    "up_left": [CAM_VALUE, -CAM_VALUE],
    "down_right": [-CAM_VALUE, CAM_VALUE],
    "down_left": [-CAM_VALUE, -CAM_VALUE],
}

def move_action_to_multihot(move_action: str) -> list:
    """
    Convert move_action string to a 6-dim multihot vector.
    Supports single keys ('W') and combinations ('WA', 'WD', etc.)
    """
    vector = [0, 0, 0, 0, 0, 0]
    if not move_action:
        return vector
        
    act_up = move_action.upper()
    # Check for each key's presence in the string
    for key, index in KEY_TO_INDEX.items():
        if key in act_up:
            vector[index] = 1
            
    return vector

def view_action_to_mouse(view_action: str) -> list:
    return VIEW_ACTION_TO_MOUSE.get(view_action.lower(), [0.0, 0.0])

def majority_vote_blocks(actions: np.ndarray, block_size: int = 12) -> np.ndarray:
    """
    Apply majority vote to normalize actions within each block.
    Structure: 1 frame (standalone) + N blocks of block_size frames + remainder
    
    Example with 77 frames and block_size=12:
    77 = 1 + 12 + 12 + 12 + 12 + 12 + 12 + 4
    
    For each block, use majority vote per dimension to determine the action,
    then apply that action to all frames in the block.
    """
    result = actions.copy()
    num_frames = len(actions)
    
    if num_frames <= 1:
        return result
    
    # Dynamically calculate block boundaries
    # First frame (index 0) is kept as-is
    # Remaining frames (1 to end) are split into blocks of block_size
    start = 1
    while start < num_frames:
        end = min(start + block_size, num_frames)
        block = actions[start:end]
        
        # Majority vote per dimension
        if actions.dtype == np.float32 and len(block) > 0:
            if block.shape[1] == 6:  # keyboard (binary)
                voted = (block.sum(axis=0) > len(block) / 2).astype(np.float32)
            else:  # mouse (continuous values like 0.1, -0.1, 0.0)
                # For each dimension, find the most common value
                voted = np.zeros(block.shape[1], dtype=np.float32)
                for dim in range(block.shape[1]):
                    values, counts = np.unique(block[:, dim], return_counts=True)
                    voted[dim] = values[np.argmax(counts)]
            
            # Apply voted action to all frames in block
            result[start:end] = voted
        
        start = end
    
    return result

def process_episode(episode_dir, episode_id):
    json_path = os.path.join(episode_dir, 'mg', 'mg_actions.json')
    video_path = os.path.join(episode_dir, 'video.mp4')
    
    if not os.path.exists(json_path) or not os.path.exists(video_path):
        print(f"Skipping {episode_dir}: Missing json or video")
        return None

    # 1. Process Actions
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sorted_keys = sorted(data.keys(), key=lambda x: int(x))
    
    keyboard_list = []
    mouse_list = []
    
    for key in sorted_keys:
        move_action = data[key].get('move_action', '')
        multihot = move_action_to_multihot(move_action)
        keyboard_list.append(multihot)
        
        view_action = data[key].get('view_action', '')
        mouse_vector = view_action_to_mouse(view_action)
        mouse_list.append(mouse_vector)
    
    # Slice & Shift: keyboard[11:88], mouse[12:89] (length 77)
    sliced_keyboard = keyboard_list[11:11+77]
    sliced_mouse = mouse_list[12:12+77]
    
    # Convert to numpy arrays
    keyboard_arr = np.array(sliced_keyboard, dtype=np.float32)
    mouse_arr = np.array(sliced_mouse, dtype=np.float32)
    
    # Apply majority vote to normalize actions within each block
    keyboard_arr = majority_vote_blocks(keyboard_arr)
    mouse_arr = majority_vote_blocks(mouse_arr)
    
    action_dict = {
        'keyboard': keyboard_arr,
        'mouse': mouse_arr
    }
    
    # 2. File Naming
    filename_prefix = f"{episode_id:06d}"
    output_video_rel = f"{filename_prefix}.mp4"
    output_action_rel = f"{filename_prefix}_action.npy"
    output_image_rel = f"{filename_prefix}.jpg"
    
    output_video_full = os.path.join(VIDEO_OUTPUT_DIR, output_video_rel)
    output_action_full = os.path.join(VIDEO_OUTPUT_DIR, output_action_rel)
    
    # 3. Video Slicing (frames 11-87)
    ffmpeg_video_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f"select='between(n,11,87)'",
        '-vsync', 'vfr',
        output_video_full
    ]
    
    try:
        # Save Video
        subprocess.run(ffmpeg_video_cmd, check=True, capture_output=True)
        # Save Actions
        np.save(output_action_full, action_dict)
        
        return {
            "path": output_video_rel,
            "action_path": output_action_rel,
            "image_path": output_image_rel,
            "episode_id": episode_id,
            "video_path": video_path,  # Keep original for later image extraction
            "keyboard": action_dict['keyboard'].tolist(),
            "mouse": action_dict['mouse'].tolist()
        }
    except subprocess.CalledProcessError as e:
        print(f"Error processing {episode_dir}: {e.stderr.decode()}")
        return None

def extract_validation_image(video_path, output_image_full):
    """Extract validation image from original video."""
    ffmpeg_img_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', "select='eq(n,11)'",
        '-vframes', '1',
        output_image_full
    ]
    try:
        subprocess.run(ffmpeg_img_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting image from {video_path}: {e.stderr.decode()}")
        return False

# Execution
episode_dirs = sorted(glob.glob(os.path.join(DATA_DIR, 'episode_*')))
results = []

print(f"Found {len(episode_dirs)} episodes. Processing with {NUM_WORKERS} workers...")

# process all episodes in parallel (video + actions only)
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    future_to_ep = {
        executor.submit(process_episode, ep_dir, i): (i, ep_dir)
        for i, ep_dir in enumerate(episode_dirs)
    }
    
    for future in as_completed(future_to_ep):
        i, ep_dir = future_to_ep[future]
        try:
            result = future.result()
            if result:
                results.append(result)
                print(f"[{len(results)}/{len(episode_dirs)}] Processed {ep_dir}")
        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")

# Sort results by episode_id to maintain order
results.sort(key=lambda x: x['episode_id'])

# Build data structures
v2c_data = []
meta_data = []
validation_data = []

for result in results:
    # video2caption.json format
    v2c_data.append({
        "path": result["path"],
        "cap": [""],
        "action_path": result["action_path"],
        "resolution": {"width": 640, "height": 352},
        "num_frames": 77,
        "fps": 25,
        "duration": 3.08
    })
    
    # metadata.json format
    meta_data.append({
        "video_path": f"videos/{result['path']}",
        "action_path": f"videos/{result['action_path']}",
        "num_frames": 77,
        "width": 640,
        "height": 352,
        "episode_id": result["episode_id"]
    })

# extract images only for first 8 validation items
print(f"\nExtracting validation images for first 8 episodes...")
for result in results[:8]:
    output_image_full = os.path.join(VALIDATE_IMG_DIR, result['image_path'])
    if extract_validation_image(result['video_path'], output_image_full):
        # Round values to fix float precision issues
        rounded_keyboard = [[round(val, 2) for val in frame] for frame in result["keyboard"]]
        rounded_mouse = [[round(val, 2) for val in frame] for frame in result["mouse"]]

        validation_data.append({
            "caption": str(result["episode_id"]),
            "image_path": f"../../../../car_8/validate/{result['image_path']}",
            "video_path": None,
            "num_inference_steps": 40,
            "height": 352,
            "width": 640,
            "num_frames": 77,
            "keyboard_cond": rounded_keyboard,
            "mouse_cond": rounded_mouse
        })
        print(f"  Extracted validation image: {result['image_path']}")

# Write JSON files (inside car_8)
with open(os.path.join(BASE_OUTPUT_DIR, 'video2caption.json'), 'w') as f:
    json.dump(v2c_data, f, indent=4)

with open(os.path.join(BASE_OUTPUT_DIR, 'metadata.json'), 'w') as f:
    json.dump(meta_data, f, indent=4)

# Write validation.json
with open(os.path.join(BASE_OUTPUT_DIR, 'validation.json'), 'w') as f:
    json.dump({"data": validation_data}, f, indent=4)

# Write merge.txt (inside car_8)
with open(os.path.join(BASE_OUTPUT_DIR, 'merge.txt'), 'w') as f:
    f.write(f"car_8/videos,car_8/video2caption.json\n")

print(f"\nProcessing complete. Total processed: {len(results)}")
print(f"Validation images: {len(validation_data)}")
print(f"Files generated in {BASE_OUTPUT_DIR}/: videos/, validate/, video2caption.json, metadata.json, validation.json, merge.txt")
