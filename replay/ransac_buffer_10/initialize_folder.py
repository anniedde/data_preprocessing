import os
import json
import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb", help="celeb name")
    args = parser.parse_args()

    celeb = args.celeb

    src_folder = os.path.join('/playpen-nas-ssd/awang/data/mystyle', celeb)
    dest_folder = os.path.join('/playpen-nas-ssd/awang/data/replay/ransac_buffer_10', celeb)

    if os.path.exists(dest_folder):
        os.system('rm -rf ' + dest_folder)

    # copy src_folder recursively to dest_folder
    shutil.copytree(src_folder, dest_folder)

    for video in range(0, 10):
        video_folder = os.path.join(dest_folder, str(video), 'train')
        subfolders = [f.path for f in os.scandir(video_folder) if f.is_dir()]

        for subfolder in subfolders:
            # rename all files in the subfolder
            for file in os.listdir(subfolder):
                if file.endswith('.pt') or file.endswith('.png'):
                    os.rename(os.path.join(subfolder, file), os.path.join(subfolder, f'{video}_{file}'))