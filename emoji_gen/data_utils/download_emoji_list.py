# credit to Evan Zhou
# https://github.com/EvanZhouDev/open-genmoji.git

import json
import requests
from pathlib import Path
import glob
from PIL import Image
from io import BytesIO

def create_white_background_image(image_data):
    img = Image.open(BytesIO(image_data))
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, "WHITE")
        background.paste(img, mask=img)
        return background
    return img

def setup_folders():
    data_dir = Path("data")
    emoji_dir = data_dir / "emoji"
    raw_dir = data_dir / "raw"
    data_dir.mkdir(exist_ok=True)
    emoji_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    return data_dir, emoji_dir, raw_dir

def get_next_number(emoji_dir):
    existing_files = glob.glob(str(emoji_dir / "img*.png"))
    if not existing_files:
        return 1
    numbers = [int(f.split("img")[-1].split(".")[0]) for f in existing_files]
    return max(numbers) + 1

def download_emojis():
    data_dir, emoji_dir, raw_dir = setup_folders()
    start_num = get_next_number(emoji_dir)

    with open(data_dir / "emojisPruned.json", "r") as f:
        emoji_data = json.load(f)

    total = len(emoji_data)

    for i, emoji in enumerate(emoji_data[start_num - 1:], start=start_num):
        print(f"📥 Downloading {i}/{total}: {emoji['processed']}")
        try:
            response = requests.get(emoji["link"], timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to download {emoji['processed']}: {e}")
            continue

        raw_path = raw_dir / f"{emoji['name']}.png"
        with open(raw_path, "wb") as f:
            f.write(response.content)

        img_with_bg = create_white_background_image(response.content)
        img_with_bg.save(emoji_dir / f"img{i}.png", "PNG")

        with open(emoji_dir / f"img{i}.txt", "w") as f:
            f.write(emoji["processed"])