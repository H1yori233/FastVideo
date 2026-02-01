import json
import os
import sys
import numpy as np
import glob
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
VIDEO_SOURCE_DIR = 'mc_32k/videos'
BASE_OUTPUT_DIR = 'mc_32k_gen'
GEN_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, 'images')
VALIDATE_IMG_DIR = os.path.join(BASE_OUTPUT_DIR, 'validate')
NUM_WORKERS = 128  # Number of parallel workers
os.makedirs(GEN_DATA_DIR, exist_ok=True)
os.makedirs(VALIDATE_IMG_DIR, exist_ok=True)

# Slicing Configuration
FRAME_COUNT = 77
BLOCK_SIZE = 12

# Action Mapping Configuration
KEY_TO_INDEX = {'W': 0, 'D': 1, 'A': 2, 'S': 3}
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

def process_single_sample(args):
    i, src_video = args
    prefix = f"gen_{i:06d}"
    img_rel = f"{prefix}.jpg"
    action_rel = f"{prefix}_action.npy"
    
    img_full = os.path.join(GEN_DATA_DIR, img_rel)
    action_full = os.path.join(GEN_DATA_DIR, action_rel)
    
    # 1. Extract first frame
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', src_video,
        '-vf', "select='eq(n,0)'", '-vframes', '1',
        img_full
    ]
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
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        return None

def main():
    video_files = glob.glob(os.path.join(VIDEO_SOURCE_DIR, "*.mp4"))
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

    # Write output files
    v2c_path = os.path.join(BASE_OUTPUT_DIR, "video2caption.json")
    # Store minimal data in video2caption
    v2c_data = []
    for res in results:
        v2c_data.append({
            "path": res["path"],
            "action_path": res["action_path"],
            "cap": res["cap"],
            "resolution": res["resolution"],
            "num_frames": res["num_frames"],
            "fps": res["fps"],
            "duration": res["duration"]
        })

    with open(v2c_path, 'w') as f:
        json.dump(v2c_data, f, indent=4)
        
    validation_data = []
    print(f"\nCreating validation.json for first 8 episodes...")
    for res in results[:8]:
        # Copy image to validation dir for preview consistency
        val_img_path = os.path.join(VALIDATE_IMG_DIR, os.path.basename(res["path"]))
        src_img_path = os.path.join(BASE_OUTPUT_DIR, res["path"])
        import shutil
        shutil.copy(src_img_path, val_img_path)

        validation_data.append({
            "caption": str(res["index"]),
            "image_path": f"../../../../{BASE_OUTPUT_DIR}/validate/{os.path.basename(res['path'])}",
            "video_path": None,
            "num_inference_steps": 40,
            "height": 352,
            "width": 640,
            "num_frames": FRAME_COUNT,
            "keyboard_cond": [[round(v, 2) for v in frame] for frame in res["keyboard"]],
            "mouse_cond": [[round(v, 2) for v in frame] for frame in res["mouse"]]
        })

    with open(os.path.join(BASE_OUTPUT_DIR, 'validation.json'), 'w') as f:
        json.dump({"data": validation_data}, f, indent=4)
        
    merge_path = os.path.join(BASE_OUTPUT_DIR, "merge.txt")
    with open(merge_path, 'w') as f:
        # Format: <folder_path>,<json_file_path>
        f.write(f"{BASE_OUTPUT_DIR},{BASE_OUTPUT_DIR}/video2caption.json\n")

    print(f"\nGeneration complete. Files saved in {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
