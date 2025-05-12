import os
import shutil

# Define the source and destination directories
source_dir = "/playpen-nas-ssd/awang/data/eg3d"
destination_dir = "/playpen-nas-ssd/awang/data/mystyle"

# Define the list of celebs
celebs = ['Harry', 'IU', 'Jennie', 'Margot']

# Iterate over each celeb
for celeb in celebs:
    for video in os.listdir(os.path.join(source_dir, celeb)):
        if not os.path.isdir(os.path.join(source_dir, celeb, video)):
            continue
        if 'train' not in os.listdir(os.path.join(source_dir, celeb, video)):
            continue
        for split in ["train", "test"]:
            source_path = os.path.join(source_dir, celeb, video, split)
            print(f'Processing {source_path}')
            destination_path = os.path.join(destination_dir, celeb, video, split, 'raw')
            os.makedirs(destination_path, exist_ok=True)
            for image in os.listdir(source_path):
                if image.endswith(".png") and not os.path.exists(os.path.join(destination_path, image)):
                    shutil.copyfile(os.path.join(source_path, image), os.path.join(destination_path, image))
                    print(f'Copied {image} from {source_path} to {destination_path}')