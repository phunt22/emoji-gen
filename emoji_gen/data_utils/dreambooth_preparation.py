# credit to Evan Zhou
# https://github.com/EvanZhouDev/open-genmoji.git

import json
import requests
from pathlib import Path
import glob
from PIL import Image
from io import BytesIO
import random
import logging
import shutil
from emoji_gen.config import (
    TEST_COUNT, INSTANCE_COUNT,
    DATA_SPLIT_SEED,
    EMOJI_DATA_PATH,
    TRAIN_DATA_PATH,     
    VAL_DATA_PATH,      
    TEST_DATA_PATH_IMAGES, 
    TEST_METADATA_PATH,
    MAX_CLASS_IMAGES,
    RAG_DATA_PATH
)

logger = logging.getLogger(__name__)


# directories
_DATA_DIR = Path("data") # Base data directory for downloads
_EMOJI_OUTPUT_DIR = _DATA_DIR / "emoji"
_RAW_OUTPUT_DIR = _DATA_DIR / "raw"
_EMOJIS_PRUNED_JSON_PATH = _DATA_DIR / "emojisPruned.json"

def _create_white_background_image(image_data):
    try:
        img = Image.open(BytesIO(image_data))
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, "WHITE")
            background.paste(img, mask=img)
            img = background.convert("RGB")
        elif img.mode == "LA":
            img = img.convert("RGBA") # Convert to RGBA first
            background = Image.new("RGBA", img.size, "WHITE")
            background.paste(img, mask=img)
            img = background.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Error in _create_white_background_image: {e}")
        return None

def _setup_emoji_download_folders():
    _DATA_DIR.mkdir(exist_ok=True)
    _EMOJI_OUTPUT_DIR.mkdir(exist_ok=True)
    _RAW_OUTPUT_DIR.mkdir(exist_ok=True)
    return _EMOJI_OUTPUT_DIR, _RAW_OUTPUT_DIR

def _get_next_emoji_file_number(emoji_dir: Path) -> int:
    existing_files = glob.glob(str(emoji_dir / "img*.png"))
    if not existing_files:
        return 1
    numbers = []
    for f_path_str in existing_files:
        f_path = Path(f_path_str)
        try:
            num_str = f_path.stem.replace("img", "")
            if num_str.isdigit():
                numbers.append(int(num_str))
        except ValueError:
            logger.warning(f"Could not parse number from filename: {f_path.name} in _get_next_emoji_file_number")
    return max(numbers) + 1 if numbers else 1

def _ensure_emojis_downloaded_and_processed():
    logger.info(f"Ensuring emojis are downloaded and processed into {_EMOJI_OUTPUT_DIR}")
    print(f"Ensuring emojis are downloaded and processed into {_EMOJI_OUTPUT_DIR}...")
    
    emoji_dir, raw_dir = _setup_emoji_download_folders()

    if not _EMOJIS_PRUNED_JSON_PATH.exists():
        msg = f"{_EMOJIS_PRUNED_JSON_PATH} not found. Required for downloading emojis. Please run previous preparation steps."
        logger.error(msg)
        print(f"ERROR: {msg}")
        return 

    with open(_EMOJIS_PRUNED_JSON_PATH, "r", encoding='utf-8') as f:
        emoji_data_list = json.load(f)

    total_in_json = len(emoji_data_list)
    if not total_in_json:
        logger.warning(f"{_EMOJIS_PRUNED_JSON_PATH} is empty. No emojis to download.")
        print(f"Warning: {_EMOJIS_PRUNED_JSON_PATH} is empty. No new emojis to download.")
        return

    # track which number to name the file
    processed_count = 0
    for i, emoji_info in enumerate(emoji_data_list):
        current_file_num = i + 1 
        img_filename = f"img{current_file_num}.png"
        txt_filename = f"img{current_file_num}.txt"

        output_image_path = emoji_dir / img_filename
        output_text_path = emoji_dir / txt_filename

        if output_image_path.exists() and output_text_path.exists():
            processed_count +=1
            if current_file_num % 100 == 0 or current_file_num == total_in_json:
                logger.debug(f"Skipping {emoji_info['name']} ({img_filename}), already exists.")
            continue 

        logger.info(f"Downloading/Processing {current_file_num}/{total_in_json}: {emoji_info['processed']} - {emoji_info['link']}")
        print(f"Downloading/Processing {current_file_num}/{total_in_json}: {emoji_info['processed']}")

        try:
            response = requests.get(emoji_info["link"], timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {emoji_info['processed']} from {emoji_info['link']}: {e}")
            print(f"Download failed for {emoji_info['processed']}. Skipping.")
            continue

        # to raw dir
        raw_img_path = raw_dir / f"{emoji_info['name']}.png"
        try:
            with open(raw_img_path, "wb") as f_raw:
                f_raw.write(response.content)
        except IOError as e:
            logger.error(f"Failed to save raw image {raw_img_path}: {e}")

        # save to emoji dir with white backgound
        processed_image = _create_white_background_image(response.content)
        if processed_image:
            try:
                processed_image.save(output_image_path, "PNG")
                with open(output_text_path, "w", encoding='utf-8') as f_text:
                    f_text.write(emoji_info["processed"]) # Save the 'processed' name as caption
                processed_count += 1
            except IOError as e:
                logger.error(f"Failed to save processed image/text for {emoji_info['name']} ({output_image_path}): {e}")
            except Exception as e_pil:
                 logger.error(f"PIL error processing/saving for {emoji_info['name']} ({output_image_path}): {e_pil}")
        else:
            logger.warning(f"Image processing failed for {emoji_info['name']}. Skipping save of {img_filename}.")
            print(f"Image processing failed for {emoji_info['name']}. Skipping save.")

    logger.info(f"Finished emoji download and processing. {processed_count}/{total_in_json} emojis are present/processed in {emoji_dir}.")
    print(f"Finished emoji download and processing. {processed_count}/{total_in_json} emojis are in {emoji_dir}.")

def setup_folders() -> tuple[Path, Path, Path, Path]:

    class_images_dir = Path(TRAIN_DATA_PATH)  # "class_images"
    instance_images_dir = Path(VAL_DATA_PATH)  # "instance_images"
    test_images_dir = Path(TEST_DATA_PATH_IMAGES)  # "test_images"
    rag_images_dir = Path(RAG_DATA_PATH)  # "rag_images"
    
    for dir_path in [class_images_dir, instance_images_dir, test_images_dir, rag_images_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return class_images_dir, instance_images_dir, test_images_dir, rag_images_dir

def find_existing_emojis():
    data_root = Path(EMOJI_DATA_PATH).parent  
    emoji_dir = data_root / "emoji"  
    
    if not emoji_dir.exists():
        raise FileNotFoundError(
            f"Emoji dir not found. Run data_utils scripts"
        ) 
    image_files = sorted(emoji_dir.glob("img*.png"))
    if not image_files:
        raise FileNotFoundError(
            f"No emoji images found in {emoji_dir}"
        )
    return emoji_dir, image_files

def load_emoji_metadata_with_images(emoji_dir, image_files):
    emoji_data_json_path = Path(EMOJI_DATA_PATH) 
    if not emoji_data_json_path.exists():
        raise FileNotFoundError(
            f"{emoji_data_json_path} not found"
        )
    
    with open(emoji_data_json_path, "r") as f:
        all_emoji_data = json.load(f)
    matched_data = []
    for i, emoji_info in enumerate(all_emoji_data):
        if i < len(image_files):
            emoji_with_path = emoji_info.copy()
            emoji_with_path['local_image_path'] = str(image_files[i])
            txt_file = emoji_dir / f"img{i+1}.txt"
            if txt_file.exists():
                try:
                    with open(txt_file, 'r') as f:
                        emoji_with_path['local_caption'] = f.read().strip()
                except Exception as e:
                    logger.warning(f"Could not read caption file {txt_file}: {e}")
            matched_data.append(emoji_with_path)
        else:
            # dont access past the end of len
            break
    return matched_data

def organize_emojis():
    _ensure_emojis_downloaded_and_processed()

    class_images_dir, instance_images_dir, test_images_dir, rag_images_dir = setup_folders()
    
    emoji_dir, image_files = find_existing_emojis()
    emoji_data_with_images = load_emoji_metadata_with_images(emoji_dir, image_files)
    
    if not emoji_data_with_images:
        logger.error("No emoji data loaded (emoji_data_with_images is empty). Cannot proceed.")
        return

    min_needed_for_splits = INSTANCE_COUNT + TEST_COUNT
    if len(emoji_data_with_images) < min_needed_for_splits:
        logger.warning(f"Not enough emojis for full DreamBooth splits. Have {len(emoji_data_with_images)}, need {min_needed_for_splits}")

    temp_shuffled_data_for_splits = list(emoji_data_with_images) 
    random.seed(DATA_SPLIT_SEED)
    random.shuffle(temp_shuffled_data_for_splits)
    
    class_count = min(max(len(temp_shuffled_data_for_splits) - INSTANCE_COUNT - TEST_COUNT, 0), MAX_CLASS_IMAGES)
    actual_data_length_for_splits = len(temp_shuffled_data_for_splits)
    class_data = temp_shuffled_data_for_splits[:min(class_count, actual_data_length_for_splits)]
    
    instance_start_idx = min(class_count, actual_data_length_for_splits)
    instance_end_idx = min(instance_start_idx + INSTANCE_COUNT, actual_data_length_for_splits)
    instance_data = temp_shuffled_data_for_splits[instance_start_idx:instance_end_idx]
    
    test_start_idx = instance_end_idx
    test_end_idx = min(test_start_idx + TEST_COUNT, actual_data_length_for_splits)
    test_data = temp_shuffled_data_for_splits[test_start_idx:test_end_idx]
    
    total_for_splits = len(class_data) + len(instance_data) + len(test_data)

    print(f"📊 Organizing emoji data for DreamBooth splits:")
    print(f"   📚 Class images: {len(class_data)} images")
    print(f"   🔍 Instance images: {len(instance_data)} images") 
    print(f"   🎯 Test images: {len(test_data)} images")
    print(f"   💡 Using {total_for_splits}/{len(emoji_data_with_images)} available images for splits.")
    
    print(f"\n📁 Copying class images...")
    copy_images(class_data, class_images_dir)
    
    print(f"📁 Copying instance images...")
    copy_images(instance_data, instance_images_dir)
    
    print(f"📁 Copying test images and creating metadata...")
    copy_images_with_metadata(test_data, test_images_dir)

    logger.info(f"\n🖼️ Populating RAG image directory at {rag_images_dir} with all {len(emoji_data_with_images)} pruned emojis...")
    copy_images(emoji_data_with_images, rag_images_dir, preserve_original_names=True)


def copy_images(emoji_list, output_dir, preserve_original_names=False):
    successful_copies = 0
    missing_sources = 0
    for idx, emoji_info in enumerate(emoji_list, 1):
        local_image_path_str = emoji_info.get('local_image_path')
        link = emoji_info.get('link') ## to get the original names

        if not local_image_path_str:
            logger.warning(f"Skipping copy for {emoji_info.get('name', 'Unknown emoji')} due to missing 'local_image_path'.")
            missing_sources += 1
            continue
        
        local_image_path = Path(local_image_path_str)
        if not local_image_path.exists():
            logger.warning(f"Source image not found: {local_image_path}. Emoji: {emoji_info.get('name')}")


            if preserve_original_names and link:
                try:
                    logger.info(f"Attempting to download missing image for RAG: {link}")
                    response = requests.get(link, timeout=10)
                    response.raise_for_status()
                    target_filename = Path(link).name
                    output_file_path = output_dir / target_filename
                    with open(output_file_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Successfully downloaded {target_filename} to {output_dir}")
                    successful_copies += 1
                except Exception as e_download:
                    logger.error(f"Failed to download {link}: {e_download}")
                    missing_sources += 1
            else:
                missing_sources += 1
            continue

        if preserve_original_names:
            print(f"Preserving original names: {link}")
            if not link:
                logger.warning(f"Cannot preserve original name for {local_image_path} as 'link' is missing. Skipping.")
                missing_sources += 1
                continue
            output_filename = Path(link).name
        else:
            output_filename = f"emoji_{idx:03d}.png"
        
        output_path = output_dir / output_filename


        # gracefully continue if a copy fails
        try:
            shutil.copy2(local_image_path, output_path)
            successful_copies += 1
        except Exception as e:
            logger.error(f"Failed to copy {local_image_path} to {output_path}: {e}")
            missing_sources +=1 
            
    logger.info(f"Copied {successful_copies}/{len(emoji_list)} images to {output_dir}. {missing_sources} missing or failed.")

def copy_images_with_metadata(emoji_list, output_dir):
    successful_copies = 0
    test_metadata = []
    for idx, emoji_info in enumerate(emoji_list, 1):
        local_image_path = Path(emoji_info['local_image_path'])
        if not local_image_path.exists():
            logger.warning(f"Local image not found: {local_image_path}")
            continue
        try:
            output_filename = f"emoji_{idx:03d}.png"
            output_path = output_dir / output_filename
            shutil.copy2(local_image_path, output_path)

            # then add the metadata entry
            test_metadata.append({
                "image_file_name": output_filename,
                "image_path_absolute": str(output_path.resolve()),
                "instance_prompt": f"sks emoji of {emoji_info.get('name', 'an emoji')}",
                "processed": emoji_info.get('processed', emoji_info.get('local_caption', emoji_info.get('name', ''))),
                "original_emoji_name": emoji_info.get('name', ''),
                "original_link": emoji_info.get('link', '')
            })
            successful_copies += 1
        
        except Exception as e:
            logger.error(f"Failed to copy {local_image_path}: {e}")
            continue  # to not fail
    metadata_path = Path(TEST_METADATA_PATH)
    with open(metadata_path, 'w') as f:
        json.dump(test_metadata, f, indent=4)
    
    logger.info(f"Copied {successful_copies}/{len(emoji_list)} images and saved metadata")

def verify_dreambooth_structure():
    """Verify the directory structure and existence of test metadata."""
    class_images_dir = Path(TRAIN_DATA_PATH)  # "class_images"
    instance_images_dir = Path(VAL_DATA_PATH)  # "instance_images" 
    test_images_dir = Path(TEST_DATA_PATH_IMAGES)  # "test_images"
    test_meta_file = Path(TEST_METADATA_PATH)
    
    class_count = len(list(class_images_dir.glob("*.png"))) if class_images_dir.exists() else 0
    instance_count = len(list(instance_images_dir.glob("*.png"))) if instance_images_dir.exists() else 0
    test_img_count = len(list(test_images_dir.glob("*.png"))) if test_images_dir.exists() else 0
    
    logger.info(f"\n📊 Verifying DreamBooth Data Structure:")
    print(f"\n📊 Verifying DreamBooth Data Structure:")
    print(f"   📚 Class images ({class_images_dir}): {class_count}")
    print(f"   🔍 Instance images ({instance_images_dir}): {instance_count}")
    print(f"   🎯 Test images ({test_images_dir}): {test_img_count}")
    
    all_ok = True
    if class_count == 0:
        logger.error(f"❌ No class images found in {class_images_dir}. Training will likely fail.")
        print(f"❌ No class images found in {class_images_dir}. Training will likely fail.")
        all_ok = False
    
    if instance_count == 0:
        logger.warning(f"No instance images found in {instance_images_dir}. Instance training might be skipped.")
        print(f"No instance images found in {instance_images_dir}. Instance training might be skipped.")
        
    if test_img_count == 0:
        logger.warning(f"No test images found in {test_images_dir}. Evaluation will not be possible.")
        print(f"No test images found in {test_images_dir}. Evaluation will not be possible.")
        
    if not test_meta_file.exists() and test_img_count > 0:
        logger.warning(f"Test images found in {test_images_dir}, but test metadata file {test_meta_file} is missing.")
        print(f"Test images found in {test_images_dir}, but test metadata file {test_meta_file} is missing.")
    elif test_meta_file.exists():
        print(f"   💾 Test metadata ({test_meta_file}): Found")
    else:
        print(f"   💾 Test metadata ({test_meta_file}): Not found (expected if no test images).")
        
    if all_ok and class_count > 0:
        print(f"✅ DreamBooth data structure is ready!")
        print(f"   📁 class_images: {class_count} images")
        print(f"   📁 instance_images: {instance_count} images")
        print(f"   📁 test_images: {test_img_count} images")
    elif not all_ok:
        print(f"❌ Issues found in data structure. Please check logs.")
         
    return all_ok and class_count > 0 ## make sure that we have the class

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting direct test of dreambooth_preparation.py...")
    
    organize_emojis()
    verify_dreambooth_structure()