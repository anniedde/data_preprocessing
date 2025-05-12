import argparse
import os
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a folder.')
    parser.add_argument('--folder', type=str, help='Path to the folder', required=True)

    args = parser.parse_args()

    # Access the folder argument
    folder_name = args.folder

    dataset = {}
    dataset['labels'] = []

    files = sorted(os.listdir(folder_name))
    for file in files:
        if file.endswith('.png'):
            cam_param_file = file.replace('.png', '.npy')
            cam_param = np.load(os.path.join(folder_name, cam_param_file))
            cam_param = cam_param.tolist()
            dataset['labels'].append([file, cam_param])

    save_loc = os.path.join(folder_name, 'dataset.json')
    with open(save_loc, 'w') as f:
        json.dump(dataset, f)