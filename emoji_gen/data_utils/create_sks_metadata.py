import json
import logging
from pathlib import Path
import glob
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMOJI_IMAGE_DIR = DATA_DIR / "emoji"
OUTPUT_METADATA_FILE = EMOJI_IMAGE_DIR / "metadata.jsonl" 

INSTANCE_PROMPT_PREFIX = "an sks emoji of"

def create_metadata_jsonl():

    if not EMOJI_IMAGE_DIR.exists():
        print(f"Error: Source directory {EMOJI_IMAGE_DIR} not found.")
        return

    image_files = sorted(EMOJI_IMAGE_DIR.glob("img*.png"))

    if not image_files:
        print(f"Warning: No image files found in {EMOJI_IMAGE_DIR}. metadata.jsonl will be empty or not created.")
        return

    created_lines = 0
    skipped_entries = 0
    processed_txt_files = []

    with open(OUTPUT_METADATA_FILE, 'w', encoding='utf-8') as outfile:
        for img_path in image_files:
            base_name = img_path.stem # e.g., "img1"
            txt_filename = base_name + ".txt"
            txt_path = EMOJI_IMAGE_DIR / txt_filename

            if not txt_path.exists():
                skipped_entries += 1
                continue

            try:
                with open(txt_path, 'r', encoding='utf-8') as caption_file:
                    caption = caption_file.read().strip()
                
                if not caption:
                    skipped_entries += 1
                    continue
                
                file_name_field = img_path.name
                
                text_field = f"{INSTANCE_PROMPT_PREFIX} {caption}"

                metadata_entry = {
                    "file_name": file_name_field,
                    "text": text_field
                }
                
                outfile.write(json.dumps(metadata_entry) + '\n')
                created_lines += 1
                processed_txt_files.append(txt_path)

            except Exception as e:
                print(f"Error processing image {img_path} or caption {txt_path}: {e}")
                skipped_entries += 1
    
    if created_lines > 0:
        print(f"Successfully created {OUTPUT_METADATA_FILE} with {created_lines} entries.")
        
        deleted_txt_count = 0
        print(f"Attempting to delete {len(processed_txt_files)} processed .txt caption files...")
        for txt_file_to_delete in processed_txt_files:
            try:
                os.remove(txt_file_to_delete)
                deleted_txt_count += 1
            except OSError as e_os:
                print(f"Error deleting file {txt_file_to_delete}: {e_os}")
        print(f"Successfully deleted {deleted_txt_count}/{len(processed_txt_files)} .txt files.")

    else:
        print(f"Warning: No entries written to {OUTPUT_METADATA_FILE}. {skipped_entries} entries skipped.")

    if skipped_entries > 0:
        print(f"{skipped_entries} entries were skipped.")


if __name__ == "__main__":
    print(f"Starting metadata.jsonl creation process...")
    print(f"Reading from: {EMOJI_IMAGE_DIR}")
    print(f"Writing to: {OUTPUT_METADATA_FILE}")
    create_metadata_jsonl()