import os
import shutil
import numpy as np
import json
import argparse
from itertools import combinations
import torch
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, LinearConstraint
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
from collections import Counter
from random import shuffle
import random

root_dir = os.path.dirname(os.path.realpath(__file__))

def initialize_folder(celeb):
    src_folder = os.path.join('/playpen-nas-ssd/awang/data/mystyle', celeb)
    dest_folder = os.path.join(root_dir, celeb)

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


def extract_latents(folder):
    anchor_folder = os.path.join(folder, 'anchors')
    latent_files = os.listdir(anchor_folder)
    ret_list = []

    for i in range(len(latent_files)):
        vid_num = int(latent_files[i].split('_')[0])
        file = latent_files[i]
        latent = torch.load(os.path.join(anchor_folder, file)).cpu().numpy().flatten()
        
        ret_list.append((file.split('.')[0], latent, vid_num))

    return ret_list

def get_all_latents(celeb, folder):
    prev_folder = os.path.join(root_dir, celeb, str(int(folder) - 1), 'train')
    replay_folder = os.path.join(root_dir, celeb, str(int(folder) - 1), 'replay')

    latent_list = extract_latents(prev_folder)

    if os.path.exists(replay_folder):
        latent_list += extract_latents(replay_folder)

    return latent_list

def select_indices(latents, buffer_size):
    selected_indices = []

    unique_vids = set([latent[2] for latent in latents])

    # create a dict mapping vid_num to indices corresponding to that vid_num
    vid_to_indices = {}
    for i, latent in enumerate(latents):
        vid_num = latent[2]
        if vid_num not in vid_to_indices:
            vid_to_indices[vid_num] = []
        vid_to_indices[vid_num].append(i)
    
    target_num_unique_vids = min(buffer_size, len(unique_vids))
    
    if buffer_size <= len(unique_vids):
        vids = random.sample(unique_vids, target_num_unique_vids)
        selected_indices = []
        for vid in vids:
            # choose a random index from vid_to_indices[vid]
            selected_indices.append(random.choice(vid_to_indices[vid]))
    else:
        # choose a random ordering of the vids
        vids = list(unique_vids)
        shuffle(vids)
        i = 0
        while len(selected_indices) < buffer_size:
            j = i % len(vids)
            vid = vids[j]
            vid_indices = vid_to_indices[vid]
            shuffle(vid_indices)
            selected_indices.append(vid_indices.pop())
            if len(vid_indices) == 0:
                vids.remove(vid)
            else:
                i += 1

    return selected_indices

def get_buffer(celeb, folder, buffer_size):
    print(f'Getting constrained random size {buffer_size} buffer for {celeb} {folder}')
    latents = get_all_latents(celeb, folder)

    selected_indices = select_indices(latents, buffer_size)
    anchors_dest_folder = os.path.join(root_dir, celeb, folder, 'replay', 'anchors')
    images_dest_folder = os.path.join(root_dir, celeb, folder, 'replay', 'preprocessed')

    if os.path.exists(anchors_dest_folder):
        shutil.rmtree(anchors_dest_folder)
    os.makedirs(anchors_dest_folder)
    if os.path.exists(images_dest_folder):
        shutil.rmtree(images_dest_folder)
    os.makedirs(images_dest_folder)

    for index in selected_indices:
        vid_num = latents[index][2]
        filename = latents[index][0]

        # copy latent
        src_latent_path = os.path.join(root_dir, celeb, str(vid_num), 'train', 'anchors', f'{filename}.pt')
        dest_latent_path = os.path.join(anchors_dest_folder, f'{filename}.pt')
        shutil.copy(src_latent_path, dest_latent_path)

        # copy image
        src_image_path = os.path.join(root_dir, celeb, str(vid_num), 'train', 'preprocessed', f'{filename}.png')
        dest_image_path = os.path.join(images_dest_folder, f'{filename}.png')
        shutil.copy(src_image_path, dest_image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb", help="celeb name")
    parser.add_argument('--device', type=str, default='0', help='device number')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    initialize_folder(args.celeb)

    # Uncomment the following lines if you want to run get_buffer_ransac for each folder
    buffer_size = 10
    for i in range(1, 10):
        get_buffer(args.celeb, str(i), buffer_size)