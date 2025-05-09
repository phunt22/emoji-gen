# credit to Evan Zhou
# https://github.com/EvanZhouDev/open-genmoji.git

import json
from pathlib import Path

def main():
    data_dir = Path("data")
    
    with open(data_dir / 'emojis.json', 'r') as f:
        emoji_data = json.load(f)

    filtered = [e for e in emoji_data if 'skin-tone' not in e['name']]

    with open(data_dir / 'emojisPruned.json', 'w') as f:
        json.dump(filtered, f, indent=2)
