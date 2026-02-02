import json
import os
import sys
import numpy as np
import glob
import subprocess
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
BASE_OUTPUT_DIR = 'mc_32k'
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
NUM_WORKERS = 128  # Number of parallel workers
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'videos')
VALIDATE_IMG_DIR = os.path.join(BASE_OUTPUT_DIR, 'validate')
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(VALIDATE_IMG_DIR, exist_ok=True)

# Slicing Configuration
FRAME_START = 11
FRAME_COUNT = 77
MOUSE_SHIFT = 1 # Mouse is shifted by this amount relative to keyboard
ZERO_FIRST_FRAME = False # If True, first frame action is [0,0...] and others shift back
NUM_SHARDS = 64 # Split output into N parts

# Action Mapping Configuration
KEY_TO_INDEX = {
    'W': 0,
    'S': 1,
    'A': 2,
    'D': 3
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
    
    # Slice & Shift actions
    if ZERO_FIRST_FRAME:
        # Prepend zero to the first frame and shift others
        sliced_kb = keyboard_list[FRAME_START : FRAME_START + FRAME_COUNT - 1]
        sliced_ms = mouse_list[FRAME_START + MOUSE_SHIFT : FRAME_START + MOUSE_SHIFT + FRAME_COUNT - 1]
        
        keyboard_arr = np.zeros((FRAME_COUNT, 6), dtype=np.float32)
        mouse_arr = np.zeros((FRAME_COUNT, 2), dtype=np.float32)
        
        keyboard_arr[1:] = sliced_kb
        mouse_arr[1:] = sliced_ms
    else:
        # Standard slicing
        sliced_kb = keyboard_list[FRAME_START : FRAME_START + FRAME_COUNT]
        sliced_ms = mouse_list[FRAME_START + MOUSE_SHIFT : FRAME_START + MOUSE_SHIFT + FRAME_COUNT]
        
        keyboard_arr = np.array(sliced_kb, dtype=np.float32)
        mouse_arr = np.array(sliced_ms, dtype=np.float32)
    
    # Apply majority vote to normalize actions within each block
    # keyboard_arr = majority_vote_blocks(keyboard_arr)
    # mouse_arr = majority_vote_blocks(mouse_arr)
    
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
    
    # 3. Video Slicing
    ffmpeg_video_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f"select='between(n,{FRAME_START},{FRAME_START + FRAME_COUNT - 1})'",
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
        '-vf', f"select='eq(n,{FRAME_START})'",
        '-vframes', '1',
        output_image_full
    ]
    try:
        subprocess.run(ffmpeg_img_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting image from {video_path}: {e.stderr.decode()}")
        return False

episode_dirs = sorted([
    d for d in glob.glob(os.path.join(DATA_DIR, '**', 'episode_*'))
    if os.path.isdir(d)
])
results = []

print(f"Found {len(episode_dirs)} episodes. Processing with {NUM_WORKERS} workers...")

# process all episodes in parallel (video + actions only)
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    future_to_ep = {
        executor.submit(process_episode, ep_dir, i): (i, ep_dir)
        for i, ep_dir in enumerate(episode_dirs)
    }
    
    pbar = tqdm(total=len(episode_dirs), desc="Processing Episodes")
    for future in as_completed(future_to_ep):
        i, ep_dir = future_to_ep[future]
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception as e:
            tqdm.write(f"Error processing {ep_dir}: {e}")
        pbar.update(1)
    pbar.close()

# Sort results by episode_id to maintain order
results.sort(key=lambda x: x['episode_id'])

# extract images only for first 8 validation items
validation_data = []
print(f"\nExtracting validation images for first 8 episodes...")
for result in tqdm(results[:8], desc="Extracting Images"):
    output_image_full = os.path.join(VALIDATE_IMG_DIR, result['image_path'])
    if extract_validation_image(result['video_path'], output_image_full):
        # Round values to fix float precision issues
        rounded_keyboard = [[round(val, 2) for val in frame] for frame in result["keyboard"]]
        rounded_mouse = [[round(val, 2) for val in frame] for frame in result["mouse"]]

        validation_data.append({
            "caption": str(result["episode_id"]),
            "image_path": f"../../../../{BASE_OUTPUT_DIR}/validate/{result['image_path']}",
            "video_path": None,
            "num_inference_steps": 40,
            "height": 352,
            "width": 640,
            "num_frames": FRAME_COUNT,
            "keyboard_cond": rounded_keyboard,
            "mouse_cond": rounded_mouse
        })

# Write Sharded JSON and merge files
shard_size = math.ceil(len(results) / NUM_SHARDS)

for i in range(NUM_SHARDS):
    start_idx = i * shard_size
    end_idx = min((i + 1) * shard_size, len(results))
    shard_results = results[start_idx:end_idx]
    
    if not shard_results and i > 0:
        continue

    # Prepare data for this shard
    s_v2c = []
    s_meta = []
    for res in shard_results:
        # video2caption format
        s_v2c.append({
            "path": res["path"],
            "cap": [""],
            "action_path": res["action_path"],
            "resolution": {"width": 640, "height": 352},
            "num_frames": FRAME_COUNT,
            "fps": 25,
            "duration": round(FRAME_COUNT / 25, 2)
        })
        # metadata format
        s_meta.append({
            "video_path": f"videos/{res['path']}",
            "action_path": f"videos/{res['action_path']}",
            "num_frames": FRAME_COUNT,
            "width": 640,
            "height": 352,
            "episode_id": res["episode_id"]
        })

    # Filenames
    suffix = f"_{i}" if NUM_SHARDS > 1 else ""
    v2c_name = f"video2caption{suffix}.json"
    meta_name = f"metadata{suffix}.json"
    merge_name = f"merge{suffix}.txt"

    # Write files
    with open(os.path.join(BASE_OUTPUT_DIR, v2c_name), 'w') as f:
        json.dump(s_v2c, f, indent=4)
    with open(os.path.join(BASE_OUTPUT_DIR, meta_name), 'w') as f:
        json.dump(s_meta, f, indent=4)
    with open(os.path.join(BASE_OUTPUT_DIR, merge_name), 'w') as f:
        f.write(f"{BASE_OUTPUT_DIR}/videos,{BASE_OUTPUT_DIR}/{v2c_name}\n")

# Write validation.json (unchanged, single file)
with open(os.path.join(BASE_OUTPUT_DIR, 'validation.json'), 'w') as f:
    json.dump({"data": validation_data}, f, indent=4)

print(f"\nProcessing complete. Total processed: {len(results)}")
print(f"Validation images: {len(validation_data)}")
print(f"Shards generated: {NUM_SHARDS}")
print(f"Files generated in {BASE_OUTPUT_DIR}/: videos/, validate/, validation.json, and {NUM_SHARDS} sets of caption/meta/merge files.")
