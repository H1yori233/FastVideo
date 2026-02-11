import json
import os
import sys
import numpy as np
import glob
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import math

# Configuration
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
VIDEO_SOURCE_DIR = '../vizdoom/data/collected_data_1k'
BASE_OUTPUT_DIR = '../vizdoom/gen'
GEN_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, 'images')
VALIDATE_IMG_DIR = os.path.join(BASE_OUTPUT_DIR, 'validate')
NUM_WORKERS = 256  # Number of parallel workers
os.makedirs(GEN_DATA_DIR, exist_ok=True)
os.makedirs(VALIDATE_IMG_DIR, exist_ok=True)

# Slicing Configuration
FRAME_COUNT = 81
BLOCK_SIZE = 12
NUM_SHARDS = 64 # Split output into N parts

# Action Mapping Configuration
KEY_TO_INDEX = {'W': 0, 'S': 1, 'A': 2, 'D': 3}
CAM_VALUE = 0.1
VIEW_ACTION_TO_MOUSE = {
    "up": [CAM_VALUE, 0.0], "down": [-CAM_VALUE, 0.0],
    "left": [0.0, -CAM_VALUE], "right": [0.0, CAM_VALUE],
    "up_right": [CAM_VALUE, CAM_VALUE], "up_left": [CAM_VALUE, -CAM_VALUE],
    "down_right": [-CAM_VALUE, CAM_VALUE], "down_left": [-CAM_VALUE, -CAM_VALUE],
}
MOUSE_ACTIONS = list(VIEW_ACTION_TO_MOUSE.values()) + [[0.0, 0.0]]
KB_COMBINATIONS = ['', 'W', 'A', 'S', 'D', 'WA', 'WD', 'SA', 'SD']

def get_random_kb():
    combo = random.choice(KB_COMBINATIONS)
    vector = [0, 0, 0, 0, 0, 0]
    for char in combo:
        if char in KEY_TO_INDEX:
            vector[KEY_TO_INDEX[char]] = 1
    return vector

def generate_action_sequence(count):
    kb_arr = np.zeros((count, 6), dtype=np.float32)
    ms_arr = np.zeros((count, 2), dtype=np.float32)
    
    # Block-wise consistent generation (1 + N * BLOCK_SIZE)
    # Frame 0
    kb_arr[0] = get_random_kb()
    ms_arr[0] = random.choice(MOUSE_ACTIONS)
    
    # Remaining blocks
    start = 1
    while start < count:
        end = min(start + BLOCK_SIZE, count)
        cur_kb = get_random_kb()
        cur_ms = random.choice(MOUSE_ACTIONS)
        kb_arr[start:end] = cur_kb
        ms_arr[start:end] = cur_ms
        start = end
        
    return {'keyboard': kb_arr, 'mouse': ms_arr}

import time

def process_single_sample(args):
    i, src_video = args
    prefix = f"gen_{i:06d}"
    img_rel = f"{prefix}.jpg"
    action_rel = f"{prefix}_action.npy"
    
    img_full = os.path.join(GEN_DATA_DIR, img_rel)
    action_full = os.path.join(GEN_DATA_DIR, action_rel)
    
    # 1. Extract first frame with retry logic
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', src_video,
        '-vf', "select='eq(n,0)'", '-vframes', '1',
        img_full
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            # 2. Generate random actions
            actions = generate_action_sequence(FRAME_COUNT)
            np.save(action_full, actions)
            
            return {
                "index": i,
                "path": f"images/{img_rel}",
                "action_path": f"images/{action_rel}",
                "cap": [""],
                "resolution": {"width": 640, "height": 352},
                "num_frames": FRAME_COUNT,
                "fps": 25,
                "duration": round(FRAME_COUNT / 25, 2),
                "keyboard": actions['keyboard'].tolist(),
                "mouse": actions['mouse'].tolist()
            }
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                # Wait briefly before retrying to resolve I/O contention
                time.sleep(random.uniform(0.5, 2.0))
                continue
            else:
                print(f"Error processing sample {i} after {max_retries} attempts: {e.stderr.decode() if e.stderr else str(e)}")
                return None
        except Exception as e:
            print(f"Internal error for sample {i}: {e}")
            return None

def main():
    video_files = glob.glob(os.path.join(VIDEO_SOURCE_DIR, "**", "*.mp4"), recursive=True)
    if not video_files:
        print(f"Error: No videos found in {VIDEO_SOURCE_DIR}")
        return

    results = []
    print(f"Generating {K} samples with {NUM_WORKERS} workers...")

    tasks = [(i, random.choice(video_files)) for i in range(K)]
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_sample, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=K, desc="Generating Samples"):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: x['index'])

    # Write Sharded JSON and merge files
    shard_size = math.ceil(len(results) / NUM_SHARDS)

    print(f"\nWriting {NUM_SHARDS} shards...")
    for i in range(NUM_SHARDS):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(results))
        shard_results = results[start_idx:end_idx]
        
        if not shard_results and i > 0:
            continue

        # Prepare minimal data for video2caption
        s_v2c = []
        for res in shard_results:
            s_v2c.append({
                "path": res["path"],
                "action_path": res["action_path"],
                "cap": res["cap"],
                "resolution": res["resolution"],
                "num_frames": res["num_frames"],
                "fps": res["fps"],
                "duration": res["duration"]
            })

        # Filenames mapping to node_id * 8 + i logic
        suffix = f"_{i}" if NUM_SHARDS > 1 else ""
        v2c_name = f"video2caption{suffix}.json"
        merge_name = f"merge{suffix}.txt"

        # Write files
        with open(os.path.join(BASE_OUTPUT_DIR, v2c_name), 'w') as f:
            json.dump(s_v2c, f, indent=4)
        with open(os.path.join(BASE_OUTPUT_DIR, merge_name), 'w') as f:
            # Format: <folder_path>,<json_file_path>
            f.write(f"{BASE_OUTPUT_DIR},{BASE_OUTPUT_DIR}/{v2c_name}\n")

    print(f"\nGeneration complete. Total: {len(results)} samples.")
    print(f"Files saved in: {os.path.abspath(BASE_OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
