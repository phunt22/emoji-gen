import json
import requests
from pathlib import Path
import glob
from PIL import Image
from io import BytesIO
import random
import logging

from emoji_gen.config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, DATA_SPLIT_SEED,
    EMOJI_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,  
    TEST_DATA_PATH_IMAGES, 
    TEST_METADATA_PATH 
)

logger = logging.getLogger(__name__)

def create_white_background_image(image_data: bytes) -> Image.Image:
    """Converts an image to RGB with a white background if it's RGBA or P (palette with alpha)."""
    img = Image.open(BytesIO(image_data))
    if img.mode == "RGBA" or (img.mode == "P" and 'A' in img.info.get('transparency', b'').decode()):
        # Convert P mode with transparency to RGBA first
        if img.mode == "P":
            img = img.convert("RGBA")
        
        # Create a white background image
        background = Image.new("RGBA", img.size, (255, 255, 255, 255)) # White and opaque
        background.paste(img, (0,0), img) # Paste using alpha mask
        return background.convert("RGB") # Convert to RGB
    return img.convert("RGB") # Ensure RGB for other modes

def setup_folders() -> tuple[Path, Path, Path, Path, Path]:
    """Sets up and returns paths for data, train, validation, test, and raw image directories."""
    # Use configured paths directly
    train_dir = Path(TRAIN_DATA_PATH)
    val_dir = Path(VAL_DATA_PATH)
    test_dir = Path(TEST_DATA_PATH_IMAGES)
    
    # Raw directory for original downloads
    # Assuming EMOJI_DATA_PATH is like "data/emojisPruned.json", so its parent is "data/"
    data_root_dir = Path(EMOJI_DATA_PATH).parent 
    raw_dir = data_root_dir / "raw_downloaded_emojis" 
    
    for dir_path in [train_dir, val_dir, test_dir, raw_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")
    
    return data_root_dir, train_dir, val_dir, test_dir, raw_dir

def download_emojis(): # removed train_val_test_split arg, uses config
    data_dir_root, train_dir, val_dir, test_dir, raw_dir = setup_folders()
    
    emoji_data_json_path = Path(EMOJI_DATA_PATH)
    if not emoji_data_json_path.exists():
        logger.error(f"❌ {emoji_data_json_path} not found. Run data collection/pruning first.")
        print(f"❌ {emoji_data_json_path} not found. Run data collection/pruning first.")
        return
        
    with open(emoji_data_json_path, "r") as f:
        all_emoji_data = json.load(f)

    random.seed(DATA_SPLIT_SEED)
    random.shuffle(all_emoji_data)
    
    n = len(all_emoji_data)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    # The test set is implicitly the rest, but we explicitly define it
    # test_end = val_end + int(n * TEST_RATIO) # Should be 'n' if ratios sum to 1
    
    train_data = all_emoji_data[:train_end]
    val_data = all_emoji_data[train_end:val_end] 
    test_data = all_emoji_data[val_end:] # The remainder is the test set
    
    logger.info(f"📊 Splitting {n} emojis based on config ratios (TRAIN={TRAIN_RATIO}, VAL={VAL_RATIO}, TEST={TEST_RATIO}):")
    logger.info(f"   📚 Training: {len(train_data)} emojis")
    logger.info(f"   🔍 Validation: {len(val_data)} emojis") 
    logger.info(f"   🎯 Test (for evaluation): {len(test_data)} emojis")
    print(f"📊 Splitting {n} emojis: Training: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    # No need for "continue downloading" input, script should be idempotent or managed by higher level CLI logic.
    
    logger.info(f"\n📥 Downloading training images to {train_dir}...")
    print(f"\n📥 Downloading training images to {train_dir}...")
    download_batch(train_data, train_dir, raw_dir, "train")
    
    logger.info(f"\n📥 Downloading validation images to {val_dir}...")
    print(f"\n📥 Downloading validation images to {val_dir}...")
    download_batch(val_data, val_dir, raw_dir, "val")
    
    logger.info(f"\n📥 Downloading test images and metadata to {test_dir}...")
    print(f"\n📥 Downloading test images and metadata to {test_dir}...")
    download_batch_with_metadata(test_data, test_dir, raw_dir, "test") # Test metadata will be saved here
    
    logger.info(f"\n🎉 Download complete!")
    print(f"\n🎉 Download complete!")
    print(f"   📁 Training images: {train_dir.resolve()}")
    print(f"   📁 Validation images: {val_dir.resolve()}") 
    print(f"   📁 Test images (for evaluation): {test_dir.resolve()}")
    print(f"   💾 Test metadata (for evaluation prompts): {Path(TEST_METADATA_PATH).resolve()}")
    print(f"   📁 Raw original images: {raw_dir.resolve()}")
    
    print(f"\n📋 Data ready for DreamBooth training and evaluation.")

def download_batch(emoji_list: list, output_dir: Path, raw_dir: Path, batch_prefix: str):
    """Downloads a batch of emojis (images only) to a specific directory."""
    current_file_idx = 1 # Start numbering from 1 for each batch type (train, val)
    
    for idx, emoji_info in enumerate(emoji_list):
        emoji_log_name = emoji_info.get('processed', emoji_info.get('name', f'item_{idx}'))
        logger.info(f"({batch_prefix} batch) Downloading {idx+1}/{len(emoji_list)}: {emoji_log_name}")
        # print(f"📥 ({batch_prefix}) {idx+1}/{len(emoji_list)}: {emoji_log_name}") # Redundant with logger

        link = emoji_info.get("link")
        if not link:
            logger.warning(f"⚠️ Skipping {emoji_log_name} from {batch_prefix} batch - no link found.")
            continue

        try:
            response = requests.get(link, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download {emoji_log_name} from {link}: {e}")
            continue

        raw_filename = f"{emoji_info.get('name', 'unknown').replace(' ', '_')}_{batch_prefix}_{idx}.png"
        raw_path = raw_dir / raw_filename
        try:
            with open(raw_path, "wb") as f:
                f.write(response.content)
        except IOError as e:
            logger.error(f"❌ Failed to save raw image {raw_path}: {e}")
            continue

        try:
            img_with_bg = create_white_background_image(response.content)
            output_filename = f"emoji_{current_file_idx:03d}.png"
            final_output_path = output_dir / output_filename
            img_with_bg.save(final_output_path, "PNG")
            current_file_idx += 1
        except Exception as e:
            logger.error(f"❌ Failed to process/save image for {emoji_log_name} (from {link}): {e}")
            continue

def download_batch_with_metadata(emoji_list: list, output_dir: Path, raw_dir: Path, batch_prefix: str):
    """Downloads images and saves a metadata JSON file for the batch (typically for test set)."""
    current_file_idx = 1
    test_set_metadata = []
    
    for idx, emoji_info in enumerate(emoji_list):
        emoji_log_name = emoji_info.get('processed', emoji_info.get('name', f'item_{idx}'))
        logger.info(f"({batch_prefix} batch with metadata) Downloading {idx+1}/{len(emoji_list)}: {emoji_log_name}")
        # print(f"📥 ({batch_prefix}) {idx+1}/{len(emoji_list)}: {emoji_log_name}") # Redundant

        link = emoji_info.get("link")
        if not link:
            logger.warning(f"⚠️ Skipping {emoji_log_name} from {batch_prefix} batch - no link found.")
            continue
        
        try:
            response = requests.get(link, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download {emoji_log_name} from {link}: {e}")
            continue

        raw_filename = f"{emoji_info.get('name', 'unknown').replace(' ', '_')}_{batch_prefix}_{idx}.png"
        raw_path = raw_dir / raw_filename
        try:
            with open(raw_path, "wb") as f:
                f.write(response.content)
        except IOError as e:
            logger.error(f"❌ Failed to save raw image {raw_path}: {e}")
            continue
            
        try:
            img_with_bg = create_white_background_image(response.content)
            output_filename = f"emoji_{current_file_idx:03d}.png"
            final_output_path = output_dir / output_filename
            img_with_bg.save(final_output_path, "PNG")
            
            test_set_metadata.append({
                "image_file_name": output_filename, # Relative to the test_images directory
                "image_path_absolute": str(final_output_path.resolve()),
                "instance_prompt_used_for_training_equivalent": f"sks emoji of {emoji_info.get('name','an emoji')}", # Example
                "ground_truth_caption": emoji_info.get('processed', emoji_info.get('name', '')),
                "original_emoji_name": emoji_info.get('name', ''),
                "original_link": link,
            })
            current_file_idx += 1
        except Exception as e:
            logger.error(f"❌ Failed to process/save image for {emoji_log_name} (from {link}): {e}")
            continue
            
    metadata_output_path = Path(TEST_METADATA_PATH) # Use configured path
    try:
        with open(metadata_output_path, 'w') as f:
            json.dump(test_set_metadata, f, indent=4)
        logger.info(f"💾 Saved {batch_prefix} metadata to: {metadata_output_path}")
        print(f"💾 Saved {batch_prefix} metadata to: {metadata_output_path}")
    except IOError as e:
        logger.error(f"❌ Failed to save metadata file {metadata_output_path}: {e}")

def verify_dreambooth_structure():
    """Verify the directory structure and existence of test metadata."""
    train_dir = Path(TRAIN_DATA_PATH)
    val_dir = Path(VAL_DATA_PATH)
    test_dir = Path(TEST_DATA_PATH_IMAGES)
    test_meta_file = Path(TEST_METADATA_PATH)

    train_count = len(list(train_dir.glob("*.png"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*.png"))) if val_dir.exists() else 0
    test_img_count = len(list(test_dir.glob("*.png"))) if test_dir.exists() else 0
    
    logger.info(f"\n📊 Verifying DreamBooth Data Structure:")
    print(f"\n📊 Verifying DreamBooth Data Structure:")
    print(f"   📚 Training images ({train_dir}): {train_count}")
    print(f"   🔍 Validation images ({val_dir}): {val_count}")
    print(f"   🎯 Test images ({test_dir}): {test_img_count}")
    
    all_ok = True
    if train_count == 0:
        logger.error(f"❌ No training images found in {train_dir}. Training will likely fail.")
        print(f"❌ No training images found in {train_dir}. Training will likely fail.")
        all_ok = False
    
    if val_count == 0: # Depending on script, validation might be skipped or fail
        logger.warning(f"⚠️ No validation images found in {val_dir}. Validation steps might be skipped or fail.")
        print(f"⚠️ No validation images found in {val_dir}. Validation steps might be skipped or fail.")

    if test_img_count == 0:
        logger.warning(f"⚠️ No test images found in {test_dir}. Evaluation of the model on a holdout set will not be possible with this script's output.")
        print(f"⚠️ No test images found in {test_dir}. Evaluation of the model on a holdout set will not be possible with this script's output.")
        # Not necessarily a blocker for training, so not setting all_ok to False
        
    if not test_meta_file.exists() and test_img_count > 0:
        logger.warning(f"⚠️ Test images found in {test_dir}, but test metadata file {test_meta_file} is missing.")
        print(f"⚠️ Test images found in {test_dir}, but test metadata file {test_meta_file} is missing.")
    elif test_meta_file.exists():
         print(f"   💾 Test metadata ({test_meta_file}): Found")
    else:
         print(f"   💾 Test metadata ({test_meta_file}): Not found (expected if no test images).")


    if all_ok and train_count > 0:
        print(f"✅ Primary data structure for DreamBooth training seems OK (training images exist).")
    elif not all_ok:
         print(f"❌ Issues found in data structure. Please check logs.")
         
    return all_ok and train_count > 0 # Critical that training images exist

if __name__ == "__main__":
    # Configure basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting direct test of dreambooth_preparation.py...")
    
    # Create a dummy emojisPruned.json if it doesn't exist for testing
    dummy_pruned_path = Path(EMOJI_DATA_PATH)
    if not dummy_pruned_path.exists():
        logger.warning(f"{dummy_pruned_path} not found. Creating a dummy file for testing.")
        dummy_pruned_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_data = []
        for i in range(20): # Create 20 dummy emojis
            name = f"test emoji {i+1}"
            processed_name = f"test emoji {i+1} processed"
            # You'll need actual image links for download to work, or mock requests.
            # For a simple structural test, links aren't strictly necessary if download part is skipped/mocked.
            # Using a placeholder link that will fail but allow structure test.
            dummy_data.append({"name": name, "processed": processed_name, "link": f"https://example.com/emoji_{i+1}.png"})
        with open(dummy_pruned_path, 'w') as f:
            json.dump(dummy_data, f, indent=2)
        logger.info(f"Created dummy {dummy_pruned_path} with {len(dummy_data)} items.")

    download_emojis()
    verify_dreambooth_structure() 