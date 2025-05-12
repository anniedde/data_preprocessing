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

def get_distances(celeb):
    replay_method = root_dir.split('/')[-1]

    celeb_folder = f'/playpen-nas-ssd/awang/data/replay/{replay_method}/{celeb}'

    save_dict = {}
    for vid_num in range(10):
        anchors_folder = os.path.join(celeb_folder, str(vid_num), 'train', 'anchors')

        anchors = {}
        for anchor_file in os.listdir(anchors_folder):
            anchor = torch.load(os.path.join(anchors_folder, anchor_file)).cpu().numpy().flatten()
            anchors[anchor_file.split('.')[0]] = anchor

        # for each anchor, get the mean distance from all other anchors
        mean_distances = {}
        for anchor_file, anchor in anchors.items():
            distances = []
            for other_anchor_file, other_anchor in anchors.items():
                if anchor_file != other_anchor_file:
                    distances.append(torch.norm(torch.tensor(anchor) - torch.tensor(other_anchor)).item())
            mean_distances[anchor_file] = sum(distances) / len(distances)

        # append to save_dict
        save_dict.update(mean_distances)
    
    with open(f'/playpen-nas-ssd/awang/data/replay/{replay_method}/{celeb}/mean_distances.json', 'w') as f:
        json.dump(save_dict, f)

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

def get_all_latents(celeb, folder):
    prev_folder = os.path.join(root_dir, celeb, str(int(folder) - 1), 'train')
    replay_folder = os.path.join(root_dir, celeb, str(int(folder) - 1), 'replay')

    latent_list = extract_latents(prev_folder)

    if os.path.exists(replay_folder):
        replay_latent_list = extract_latents(replay_folder)
    else:
        replay_latent_list = []

    return latent_list, replay_latent_list

def select_indices(latents, prev_buffer_latents, buffer_size):
    selected_indices = []
    if len(prev_buffer_latents) == 0:
        # get random set of buffer_size indices
        indices = list(range(len(latents)))
        shuffle(indices)
        selected_indices = indices[:buffer_size]
        selected_files = [(latents[i][0], latents[i][2]) for i in selected_indices]
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

    selected_files = [(prev_buffer_latents[i][0], prev_buffer_latents[i][2]) for i in range(len(prev_buffer_latents))]
    return selected_files

def get_buffer(celeb, folder, buffer_size):
    print(f'Getting kmeans size {buffer_size} buffer for {celeb} {folder}')
    latents, replay_latents = get_all_latents(celeb, folder)
    distances_map = json.load(open(os.path.join(root_dir, celeb, 'mean_distances.json')))
    selected_indices = select_indices(latents, replay_latents, buffer_size)
    
    anchors_dest_folder = os.path.join(root_dir, celeb, folder, 'replay', 'anchors')
    images_dest_folder = os.path.join(root_dir, celeb, folder, 'replay', 'preprocessed')

    if os.path.exists(anchors_dest_folder):
        shutil.rmtree(anchors_dest_folder)
    os.makedirs(anchors_dest_folder)
    if os.path.exists(images_dest_folder):
        shutil.rmtree(images_dest_folder)
    os.makedirs(images_dest_folder)

    print('selected_indices: ', selected_indices)

    for (filename, vid_num) in selected_indices:
        #vid_num = latents[index][2]
        #filename = latents[index][0]

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
    parser.add_argument("--device", default='0', help="device number")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    initialize_folder(args.celeb)
    get_distances(args.celeb)

    # Uncomment the following lines if you want to run get_buffer for each folder
    buffer_size = 5
    for i in range(1, 10):
        get_buffer(args.celeb, str(i), buffer_size)