import json
import os
import random

import numpy as np

action_patterns = [
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
]

action_map = {
    tuple(action_patterns[0]): "Left  (0)",
    tuple(action_patterns[1]): "Stop  (1)",
    tuple(action_patterns[2]): "Right (2)",
    # tuple(action_patterns[0]): "Stop  (0)",
}

num_frames = 81
data_nums = 32
# data_nums = 4
dataset_root = "footsies-action"
actions_dir = os.path.join(dataset_root, "actions")
images_dir = os.path.join(dataset_root, "validate")
metadata_path = os.path.join(dataset_root, "metadata.json")
merge_path = os.path.join(dataset_root, "merge.txt")
os.makedirs(actions_dir, exist_ok=True)

for i in range(data_nums):
    sequence_indices = np.random.randint(0, len(action_patterns), size=7)
    full_sequence = []
    for action_idx in sequence_indices:
        action = action_patterns[action_idx]
        full_sequence.extend([action] * 12)

    current_len = len(full_sequence)
    if current_len > num_frames:
        full_sequence = full_sequence[:num_frames]
    else:
        full_sequence.extend([action_patterns[1]] * (num_frames - current_len))
    print(full_sequence)
    np.save(os.path.join(actions_dir, f"action_{i}.npy"), full_sequence)

jpg_files = sorted(
    f for f in os.listdir(images_dir)
    if f.lower().endswith(".jpg")
)
if not jpg_files:
    raise FileNotFoundError(f"No .jpg files found in {images_dir}")

records = []
for i in range(data_nums):
    chosen_jpg = random.choice(jpg_files)
    records.append({
        "path": f"validate/{chosen_jpg}",
        "action_path": f"actions/action_{i}.npy",
        "cap": [""],
    })

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

with open(merge_path, "w", encoding="utf-8") as f:
    f.write(f"{dataset_root},{metadata_path}\n")
