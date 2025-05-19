import json
from pathlib import Path
import random
from typing import Dict, List, Tuple

from emoji_gen.config import (
    DATA_SPLIT_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)

def split_emoji_data(
    data_path: str,
    output_dir: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    # Validate ratios
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1, got {total_ratio}")
    
    # set random seed for reproducibility
    random.seed(DATA_SPLIT_SEED)
    print(f"Using random seed: {DATA_SPLIT_SEED}")
    
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    print("Shuffling data...")
    random.shuffle(data)
    
    # split into train val test
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    # Split data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving splits to {output_dir}...")
    with open(output_dir / "train_emoji_data.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(output_dir / "val_emoji_data.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(output_dir / "test_emoji_data.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Split complete:")
    print(f"- Training set: {len(train_data)} samples")
    print(f"- Validation set: {len(val_data)} samples")
    print(f"- Test set: {len(test_data)} samples")
    
    return train_data, val_data, test_data 