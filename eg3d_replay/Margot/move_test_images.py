import os, shutil, glob

dest_folder = '/playpen-nas-ssd/awang/data/eg3d/Margot/all/test'
for vid_num in range(0, 10):
    src_folder = f'/playpen-nas-ssd/awang/data/eg3d/Margot/{vid_num}/test'
    
    # get all png images from src_folder
    src_files = os.listdir(src_folder)
    src_files = [file for file in src_files if file.endswith('.png')]

    for file in src_files:
        src_path = os.path.join(src_folder, file)
        renamed_file = f'{vid_num}_{file}'
        dest_path = os.path.join(dest_folder, renamed_file)

        shutil.copyfile(src_path, dest_path)
        print(f'Copied {file} from {src_folder} to {dest_folder}')