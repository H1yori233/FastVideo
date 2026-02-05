import os
import numpy as np

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'action')
VIDEO_OUTPUT_DIR = BASE_OUTPUT_DIR
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

FRAME_COUNT = 81
CAM_VALUE = 0.1

# Action Mapping
KEY_TO_INDEX = {
    'W': 0, 'S': 1, 'A': 2, 'D': 3,
}

VIEW_ACTION_TO_MOUSE = {
    "stop": [0.0, 0.0],
    "up": [CAM_VALUE, 0.0],
    "down": [-CAM_VALUE, 0.0],
    "left": [0.0, -CAM_VALUE],
    "right": [0.0, CAM_VALUE],
    "up_right": [CAM_VALUE, CAM_VALUE],
    "up_left": [CAM_VALUE, -CAM_VALUE],
    "down_right": [-CAM_VALUE, CAM_VALUE],
    "down_left": [-CAM_VALUE, -CAM_VALUE],
}

def get_multihot_vector(keys_str):
    """Convert string like 'WA' to [1, 0, 1, 0, 0, 0]"""
    vector = [0.0] * 6
    if not keys_str:
        return vector
    for char in keys_str.upper():
        if char in KEY_TO_INDEX:
            vector[KEY_TO_INDEX[char]] = 1.0
    return vector

def get_mouse_vector(view_str):
    """Convert view string to [x, y]"""
    return VIEW_ACTION_TO_MOUSE.get(view_str.lower(), [0.0, 0.0])

def generate_sequence(key_seq, mouse_seq):
    """
    Generates action arrays based on sequences.
    """
    keyboard_arr = np.zeros((FRAME_COUNT, 6), dtype=np.float32)
    mouse_arr = np.zeros((FRAME_COUNT, 2), dtype=np.float32)
    
    block_starts = [0, 1, 13, 25, 37, 49, 61, 73, 81]
    for block_idx in range(8):
        start = block_starts[block_idx]
        end = block_starts[block_idx + 1]
        
        k_vec = get_multihot_vector(key_seq[block_idx])
        m_vec = get_mouse_vector(mouse_seq[block_idx])
        
        keyboard_arr[start:end] = k_vec
        mouse_arr[start:end] = m_vec

    return keyboard_arr, mouse_arr

def save_action(index, keyboard_arr, mouse_arr):
    filename = f"{index:06d}_action.npy"
    filepath = os.path.join(VIDEO_OUTPUT_DIR, filename)
    
    action_dict = {
        'keyboard': keyboard_arr,
        'mouse': mouse_arr
    }
    np.save(filepath, action_dict)
    return filename

def generate_description(key_seq, mouse_seq):
    """Generates a human-readable string for the combination."""
    # Look at the first and last useful blocks to determine behavior
    k1, k_last = key_seq[1], key_seq[7]
    m1, m_last = mouse_seq[1], mouse_seq[7]
    
    # Format Keyboard Description
    if all(k == "" for k in key_seq):
        k_desc = "No Key"
    elif all(k == k1 for k in key_seq[1:]):
        k_desc = f"Hold [{k1}]"
    else:
        k_desc = f"Switch [{k1}]->[{k_last}]"
        
    # Format Mouse Description
    if all(m == "stop" for m in mouse_seq):
        m_desc = "Static"
    elif all(m == m1 for m in mouse_seq[1:]):
        m_desc = f"Hold [{m1}]"
    else:
        m_desc = f"Switch [{m1}]->[{m_last}]"
        
    return f"{k_desc} + {m_desc}"

# ==========================================
# Main Generation Logic
# ==========================================

configs = []
readme_content = []

# Group 1: Constant Keyboard, No Mouse (0-7)
keys_basic = ['W', 'S', 'A', 'D', 'WA', 'WD', 'SA', 'SD']
for k in keys_basic:
    configs.append(([k] * 8, ["stop"] * 8))

# Group 2: No Keyboard, Constant Mouse (8-15)
mouse_basic = ['up', 'down', 'left', 'right', 'up_right', 'up_left', 'down_right', 'down_left']
for m in mouse_basic:
    configs.append(([""] * 8, [m] * 8))

# Group 3: Split Keyboard, No Mouse (16-23)
split_keys = [
    ('W', 'S'), ('S', 'W'), 
    ('A', 'D'), ('D', 'A'),
    ('W', 'A'), ('W', 'D'), 
    ('S', 'A'), ('S', 'D')
]
for k1, k2 in split_keys:
    configs.append(([k1] * 4 + [k2] * 4, ["stop"] * 8))

# Group 4: No Keyboard, Split Mouse (24-31)
split_mouse = [
    ('left', 'right'), ('right', 'left'),
    ('up', 'down'), ('down', 'up'),
    ('up_left', 'up_right'), ('up_right', 'up_left'),
    ('left', 'up'), ('right', 'down')
]
for m1, m2 in split_mouse:
    configs.append(([""] * 8, [m1] * 4 + [m2] * 4))

# Group 5: Constant Keyboard + Constant Mouse (32-47)
combo_keys = ['W', 'S', 'W', 'S', 'A', 'D', 'WA', 'WD', 'W', 'S', 'W', 'S', 'A', 'D', 'WA', 'WD']
combo_mice = ['left', 'left', 'right', 'right', 'up', 'up', 'down', 'down', 'up_left', 'up_left', 'up_right', 'up_right', 'down_left', 'down_right', 'right', 'left']
for i in range(16):
    configs.append(([combo_keys[i]] * 8, [combo_mice[i]] * 8))

# Group 6: Constant Keyboard, Split Mouse (48-55)
complex_1_keys = ['W'] * 8
complex_1_mice = [
    ('left', 'right'), ('right', 'left'), 
    ('up', 'down'), ('down', 'up'),
    ('left', 'up'), ('right', 'up'),
    ('left', 'down'), ('right', 'down')
]
for i in range(8):
    m1, m2 = complex_1_mice[i]
    configs.append(([complex_1_keys[i]] * 8, [m1] * 4 + [m2] * 4))

# Group 7: Split Keyboard, Constant Mouse (56-63)
complex_2_keys_list = [
    ('W', 'S'), ('S', 'W'),
    ('A', 'D'), ('D', 'A'),
    ('W', 'A'), ('W', 'D'),
    ('S', 'A'), ('S', 'D')
]
complex_2_mouse = 'up' 
for k1, k2 in complex_2_keys_list:
    configs.append(([k1] * 4 + [k2] * 4, [complex_2_mouse] * 8))


# Execution
print(f"Preparing to generate {len(configs)} action files...")

for i, (key_seq, mouse_seq) in enumerate(configs):
    if i >= 64: break 
    
    # Generate Data
    kb_arr, ms_arr = generate_sequence(key_seq, mouse_seq)
    filename = save_action(i, kb_arr, ms_arr)
    
    # Generate Description for README
    description = generate_description(key_seq, mouse_seq)
    readme_entry = f"{i:02d}. {description}"
    readme_content.append(readme_entry)
    
    print(f"Generated {filename} -> {description}")

# Write README
readme_path = os.path.join(BASE_OUTPUT_DIR, 'README.md')
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(f"Total Files: {len(readme_content)}\n\n")
    for line in readme_content:
        f.write(line + '\n')

print(f"\nProcessing complete.")
print(f"64 .npy files generated in {VIDEO_OUTPUT_DIR}")
print(f"Manifest saved to {readme_path}")