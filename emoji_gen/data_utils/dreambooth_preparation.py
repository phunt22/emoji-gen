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
    MAX_CLASS_IMAGES
)

logger = logging.getLogger(__name__)

def setup_folders() -> tuple[Path, Path, Path]:

    class_images_dir = Path(TRAIN_DATA_PATH)  # "class_images"
    instance_images_dir = Path(VAL_DATA_PATH)  # "instance_images"
    test_images_dir = Path(TEST_DATA_PATH_IMAGES)  # "test_images"
    
    for dir_path in [class_images_dir, instance_images_dir, test_images_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return class_images_dir, instance_images_dir, test_images_dir

def find_existing_emojis():
    data_root = Path(EMOJI_DATA_PATH).parent  # data/
    emoji_dir = data_root / "emoji"  # from the download loc
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
    class_images_dir, instance_images_dir, test_images_dir = setup_folders()
    emoji_dir, image_files = find_existing_emojis()
    # this will be test data
    emoji_data_with_images = load_emoji_metadata_with_images(emoji_dir, image_files)
    min_needed = INSTANCE_COUNT + TEST_COUNT
    if len(emoji_data_with_images) < min_needed:
        raise ValueError(
            "Not enough emojis in the folder"
        )
    
    # set an upper limit on the class count, from config.py
    class_count = min(max(len(emoji_data_with_images) - INSTANCE_COUNT - TEST_COUNT, 0), MAX_CLASS_IMAGES)
    total_needed = class_count + INSTANCE_COUNT + TEST_COUNT

    random.seed(DATA_SPLIT_SEED)
    random.shuffle(emoji_data_with_images)
    needed_data = emoji_data_with_images[:total_needed]
    
    class_data = needed_data[:class_count]
    instance_data = needed_data[class_count:class_count + INSTANCE_COUNT]
    test_data = needed_data[class_count + INSTANCE_COUNT:class_count + INSTANCE_COUNT + TEST_COUNT]
    
    print(f"📊 Organizing emoji data:")
    print(f"   📚 Class images (most training data): {len(class_data)} images")
    print(f"   🔍 Instance images (50 images): {len(instance_data)} images") 
    print(f"   🎯 Test images: {len(test_data)} images")
    print(f"   💡 Using {total_needed}/{len(emoji_data_with_images)} available images")
    
    print(f"\n📁 Copying class images...")
    copy_images(class_data, class_images_dir)
    
    print(f"📁 Copying instance images...")
    copy_images(instance_data, instance_images_dir)
    
    print(f"📁 Copying test images and creating metadata...")
    copy_images_with_metadata(test_data, test_images_dir)

def copy_images(emoji_list, output_dir):
    successful_copies = 0
    for idx, emoji_info in enumerate(emoji_list, 1):
        local_image_path = Path(emoji_info['local_image_path'])
        if not local_image_path.exists():
            logger.warning(f"Local image not found: {local_image_path}")
            continue
        try:
            output_filename = f"emoji_{idx:03d}.png"
            output_path = output_dir / output_filename
            shutil.copy2(local_image_path, output_path)
            successful_copies += 1
        except Exception as e:
            logger.error(f"Failed to copy {local_image_path}: {e}")
            continue  # for final log, dont fail
    logger.info(f"Copied {successful_copies}/{len(emoji_list)} images successfully")

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
                "ground_truth_caption": emoji_info.get('processed', emoji_info.get('local_caption', emoji_info.get('name', ''))),
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