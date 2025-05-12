import os, shutil
import numpy as np
import json
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, LinearConstraint
from sklearn.decomposition import PCA
from tqdm import tqdm
from random import shuffle
import random

def prepare_folder(celeb):

    for i in range(0, 10):
        folder = os.path.join(str(i), 'train', 'preprocessed')
        # remove replay folder recursively
        if os.path.exists(os.path.join(str(i), 'train', 'replay')):
            os.system(f'rm -rf {os.path.join(str(i), "train", "replay")}')

        files = os.listdir(folder)
        for j in range(20):
            if os.path.isfile(os.path.join(folder, f'{j}.png')):
                os.rename(os.path.join(folder, f'{j}.png'), os.path.join(folder, f'{i}_{j}.png'))
                os.rename(os.path.join(folder, f'{j}_mirror.png'), os.path.join(folder, f'{i}_{j}_mirror.png'))
                os.rename(os.path.join(folder, f'{j}.npy'), os.path.join(folder, f'{i}_{j}.npy'))
                os.rename(os.path.join(folder, f'{j}_mirror.npy'), os.path.join(folder, f'{i}_{j}_mirror.npy'))
                os.rename(os.path.join(folder, f'{j}_latent.npy'), os.path.join(folder, f'{i}_{j}_latent.npy'))
                os.rename(os.path.join(folder, f'{j}_mirror_latent.npy'), os.path.join(folder, f'{i}_{j}_mirror_latent.npy'))

            #os.remove(os.path.join('all', 'train', 'preprocessed', f'{i}_{j}_latent.npy'))
            #os.remove(os.path.join('all', 'train', 'preprocessed', f'{i}_{j}_mirror_latent.npy'))
            #shutil.copy(os.path.join(folder, f'{i}_{j}_latent.npy'), os.path.join('all', 'train', 'preprocessed', f'{i}_{j}_latent.npy'))
            #shutil.copy(os.path.join(folder, f'{i}_{j}_mirror_latent.npy'), os.path.join('all', 'train', 'preprocessed', f'{i}_{j}_mirror_latent.npy'))
            #assert os.path.isfile(os.path.join(folder, f'{i}_{j}_latent.npy'))
            #assert os.path.isfile(os.path.join(folder, f'{i}_{j}_mirror_latent.npy'))

        dataset_dest_loc = os.path.join(folder, 'dataset.json')
        dataset_src_loc = os.path.join(f'/playpen-nas-ssd/awang/data/eg3d/{celeb}/', str(i), 'train', 'preprocessed', 'dataset.json')

        with open(dataset_src_loc, 'r') as f:
            dataset_src = json.load(f)

        # rename the labels in the dataset
        for entry in dataset_src['labels']:
            entry[0] = f'{i}_{entry[0]}'

        with open(dataset_dest_loc, 'w') as f:
            json.dump(dataset_src, f)

def select_indices(latents, prev_buffer_latents, buffer_size):
    selected_indices = []
    if len(prev_buffer_latents) == 0:
        # get random set of buffer_size indices
        indices = list(range(len(latents)))
        shuffle(indices)
        selected_indices = indices[:buffer_size]
        selected_files = [latents[i][0] for i in selected_indices]
        return selected_files

    for latent in latents:
        print('latent: ', latent[0])
        unique_vids = set([latent[2] for latent in prev_buffer_latents])
        # create a dict mapping vid_num to indices corresponding to that vid_num
        vid_to_indices = {}
        for i, prev_latent in enumerate(prev_buffer_latents):
            vid_num = prev_latent[2]
            if vid_num not in vid_to_indices:
                vid_to_indices[vid_num] = []
            vid_to_indices[vid_num].append(i)

        # do GDumb
        class_sizes = {}
        for vid_num in unique_vids:
            class_sizes[vid_num] = len(vid_to_indices[vid_num])
        print('class sizes: ', class_sizes)
        
        # get the class with the largest size
        max_class = max(class_sizes, key=class_sizes.get)
        # randomly select sample to replace from this class
        replaced_index = random.choice(vid_to_indices[max_class])
        # replace the sample
        prev_buffer_latents[replaced_index] = latent
        print(f'Replacing {prev_buffer_latents[replaced_index][0]} with {latent[0]}')

    selected_files = [prev_buffer_latents[i][0] for i in range(len(prev_buffer_latents))]
    return selected_files

def get_buffer(folder, buffer_size):
    prev_folders = [os.path.join(str(int(folder) - 1), 'train', 'preprocessed'), os.path.join(str(int(folder) - 1), 'train', 'replay')]

    latent_files = [file for file in os.listdir(prev_folders[0]) if file.endswith('latent.npy') and 'mirror' not in file]
    latent_list = []
    vid_nums = []
    for i in range(len(latent_files)):
        year = int(latent_files[i].split('_')[0])
        file = latent_files[i]
        #latent = np.load(os.path.join(prev_folders[0], file)).flatten()
        latent = np.load(os.path.join(prev_folders[0], file))[0]
        #latent = latent[-7:] 
        latent = latent.flatten()
        latent_list.append(latent)
        vid_nums.append(year)
    
    prev_buffer_latent_list, prev_buffer_latent_files, prev_buffer_vid_nums = [], [], []
    if os.path.exists(prev_folders[1]):
        latent_files_replay = [file for file in os.listdir(prev_folders[1]) if file.endswith('latent.npy') and 'mirror' not in file]
        for i in range(len(latent_files_replay)):
            year = int(latent_files_replay[i].split('_')[0])
            file = latent_files_replay[i]
            #latent = np.load(os.path.join(prev_folders[1], file)).flatten()
            latent = np.load(os.path.join(prev_folders[1], file))[0]
            #latent = latent[-7:]
            latent = latent.flatten()
            #latent_list.append(latent)
            #latent_files.append(file)
            #vid_nums.append(year)
            prev_buffer_latent_list.append(latent)
            prev_buffer_latent_files.append(file)
            prev_buffer_vid_nums.append(year)
    latents = list(zip(latent_files, latent_list, vid_nums))
    prev_buffer_latents = list(zip(prev_buffer_latent_files, prev_buffer_latent_list, prev_buffer_vid_nums))
    selected_latent_files = select_indices(latents, prev_buffer_latents, buffer_size)

    print('selected latents: ', selected_latent_files)
    # open dataset.json
    dataset_src_map = {}
    for prev_folder in prev_folders:
        dataset_src_loc = os.path.join(prev_folder, 'dataset.json')
        if os.path.exists(dataset_src_loc):
            with open(dataset_src_loc, 'r') as f:
                # load json
                dataset_src = json.load(f)
            
            for entry in dataset_src['labels']:
                dataset_src_map[entry[0]] = entry[1]

    dest_folder = os.path.join(folder, 'train', 'replay')
    dataset_dest_loc = os.path.join(dest_folder, 'dataset.json')
    dataset_new = {'labels': []}

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    for i, file in enumerate(selected_latent_files):
        #print(i, file)
        if file.startswith(str(int(folder) - 1)):
            shutil.copy(os.path.join(prev_folders[0], file), os.path.join(dest_folder, file))
            # also copy image
            shutil.copy(os.path.join(prev_folders[0], file.replace('_latent.npy', '.png')), os.path.join(dest_folder, file.replace("_latent.npy", ".png")))
            # also copy camera file
            shutil.copy(os.path.join(prev_folders[0], file.replace('_latent.npy', '.npy')), os.path.join(dest_folder, file.replace("_latent.npy", ".npy")))
        else:
            shutil.copy(os.path.join(prev_folders[1], file), os.path.join(dest_folder, file))
            # also copy image
            shutil.copy(os.path.join(prev_folders[1], file.replace('_latent.npy', '.png')), os.path.join(dest_folder, file.replace("_latent.npy", ".png")))
            # also copy camera file
            shutil.copy(os.path.join(prev_folders[1], file.replace('_latent.npy', '.npy')), os.path.join(dest_folder, file.replace("_latent.npy", ".npy")))
        # get the corresponding entry in dataset_0
        dataset_0_entry = dataset_src_map[file.replace('_latent.npy', '.png')]
        dataset_new['labels'].append([file.replace("_latent.npy", ".png"), dataset_0_entry])

    # save dataset_new
    with open(dataset_dest_loc, 'w') as f:
        json.dump(dataset_new, f)

if __name__ == '__main__':
    # get name of the folder that this file is in
    celeb = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    celeb = celeb.split('_')[0]
    prepare_folder(celeb)
    
    for i in range(0, 10):
        folder = os.path.join(str(i), 'train', 'preprocessed')
        # remove replay folder recursively
        if os.path.exists(os.path.join(str(i), 'train', 'replay')):
            os.system(f'rm -rf {os.path.join(str(i), "train", "replay")}')

        files = os.listdir(folder)
        for j in range(20):
            assert os.path.isfile(os.path.join(folder, f'{i}_{j}_latent.npy'))
            assert os.path.isfile(os.path.join(folder, f'{i}_{j}_mirror_latent.npy'))

        for file in os.listdir(folder):
            if file.endswith('.npy') or file.endswith('.png'):
                if not file.startswith(f'{i}_'):
                    os.remove(os.path.join(folder, file))
            
    for i in range(1, 10):
        get_buffer(str(i), 3)
    