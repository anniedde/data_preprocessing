import os
import subprocess
import numpy as np
import argparse
import json
import shutil

def convert_dataset_json(folder_path):
    loc = os.path.join(folder_path, 'dataset.json') #args.folder + '/dataset.json'
    if os.path.exists(loc):
        print('Converting dataset.json to cameras.json')
        with open(loc, "r") as f:
            dataset = json.load(f)

        cameras = {}
        for (filename, label) in dataset['labels']:
            entry = {}
            key = filename.split('.')[0]

            entry['pose'] = np.array(label[:16]).reshape(4, 4).tolist()
            entry['intrinsics'] = np.array(label[16:]).reshape(3,3).tolist()

            cameras[key] = entry

        with open(os.path.join(folder_path, 'cameras.json'), "w") as f:
            json.dump(cameras, f)

def make_folders_for_images(folder):
    image_paths = os.listdir(folder)
    for fileName in image_paths:
        if fileName.endswith('.png'):
            id_name = fileName.split('.')[0]
            image_folder = os.path.join(folder, id_name)
            if not os.path.exists(image_folder):
                os.mkdir(image_folder)
                #shutil.copy(os.path.join(folder, 'crop_1024', fileName), os.path.join(image_folder, fileName))
                shutil.copy(os.path.join(folder, fileName), os.path.join(image_folder, fileName))

def process_subfolders(folder_path):
    # get all folders that lie in folder_path recursively named test
    for root, dirs, files in os.walk(folder_path):
        if 'test' in dirs:
            test_folder = os.path.join(root, 'test', 'preprocessed')
            print('Processing folder: ' + test_folder)
            if os.path.exists(test_folder):
                convert_dataset_json(test_folder)
                make_folders_for_images(test_folder)


# Example usage
#folder_path = "/path/to/your/folder"
#process_subfolders(folder_path)

# get the folder path from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
args = parser.parse_args()

process_subfolders(args.folder)