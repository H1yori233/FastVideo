import os
import numpy as np
import glob
import sys
from tqdm import tqdm

# Configuration - matches process.py
BLOCK_SIZE = 12

def check_consistency(array, name, episode_file):
    """
    Checks if blocks of frames have identical actions.
    Pattern: frame 0 stays, then blocks of BLOCK_SIZE.
    """
    num_frames = len(array)
    if num_frames <= 1:
        return True

    all_ok = True
    start = 1
    block_idx = 1
    
    while start < num_frames:
        end = min(start + BLOCK_SIZE, num_frames)
        block = array[start:end]
        
        # Check if all frames in block are equal to the first frame of the block
        # np.all(block == block[0, :], axis=0) checks each dimension
        # We want to know if every frame in the block is a duplicate of the first one
        is_consistent = np.all(np.all(block == block[0], axis=1))
        
        if not is_consistent:
            tqdm.write(f"  [FAILED] {name} Block {block_idx} (frames {start}-{end-1}) is NOT consistent!")
            all_ok = False
        
        start = end
        block_idx += 1
        
    return all_ok

def validate(target_dir):
    print(f"Validating NPY action consistency in: {target_dir}")
    
    # Search in the videos subfolder of BASE_OUTPUT_DIR
    search_path = os.path.join(target_dir, 'videos', '*_action.npy')
    npy_files = sorted(glob.glob(search_path))
    
    if not npy_files:
        print(f"No NPY files found in {search_path}")
        return

    total_files = len(npy_files)
    passed_count = 0
    
    pbar = tqdm(npy_files, desc="Validating NPY")
    for npy in pbar:
        filename = os.path.basename(npy)
        try:
            # allow_pickle=True since it's a dict
            data = np.load(npy, allow_pickle=True).item()
            keyboard = data['keyboard']
            mouse = data['mouse']
            
            kb_ok = check_consistency(keyboard, "Keyboard", filename)
            ms_ok = check_consistency(mouse, "Mouse", filename)
            
            if kb_ok and ms_ok:
                passed_count += 1
            else:
                tqdm.write(f"  [FAIL] {filename}: Consistency check failed.")
                
        except Exception as e:
            tqdm.write(f"  [ERROR] Failed to process {filename}: {e}")

    print(f"\nSummary: {passed_count}/{total_files} files passed consistency validation.")

if __name__ == "__main__":
    # Default to 'car_8' or take from command line
    target = sys.argv[1] if len(sys.argv) > 1 else 'car_8'
    validate(target)
