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
from copy import deepcopy

def initialize_folder(celeb, root_dir, videos):
    src_folder = os.path.join('/playpen-nas-ssd/awang/data/mystyle', celeb)
    dest_folder = os.path.join(root_dir, celeb)

    if os.path.exists(dest_folder):
        os.system('rm -rf ' + dest_folder)

    # copy src_folder recursively to dest_folder
    shutil.copytree(src_folder, dest_folder)

    for video in videos:
        video_folder = os.path.join(dest_folder, str(video), 'train')
        subfolders = [f.path for f in os.scandir(video_folder) if f.is_dir()]

        for subfolder in subfolders:
            # rename all files in the subfolder
            for file in os.listdir(subfolder):
                if file.endswith('.pt') or file.endswith('.png'):
                    os.rename(os.path.join(subfolder, file), os.path.join(subfolder, f'{video}_{file}'))

def get_distance_to_convex_hull(point, hull, pca_model):
    def distance_to_point(point_guess):
        return np.linalg.norm(pca_model.transform(point.reshape(1, -1)) - point_guess)

    A = hull.equations[:, :-1]
    b = -1 * hull.equations[:, -1]
    constraint = LinearConstraint(A, -np.inf, b)
    result = minimize(distance_to_point, np.mean(hull.points, axis=0), constraints=constraint)
    closest_point = result.x
    closest_point = pca_model.inverse_transform(closest_point)
    return np.linalg.norm(point - closest_point)

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

def get_all_latents(celeb, folder, root_dir):
    prev_folder = os.path.join(root_dir, celeb, str(int(folder) - 1), 'train')
    replay_folder = os.path.join(root_dir, celeb, str(int(folder) - 1), 'replay')

    latent_list = extract_latents(prev_folder)

    if os.path.exists(replay_folder):
        latent_list += extract_latents(replay_folder)

    return latent_list

def get_buffer_ransac(celeb, folder, root_dir, buffer_size):
    print(f'Getting constrained ransac size {buffer_size} buffer for {celeb} {folder}')
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