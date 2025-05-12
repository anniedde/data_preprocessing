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
        latent_list += extract_latents(replay_folder)

    return latent_list

def select_indices(latents, distances_map, buffer_size):
    selected_indices = []

    latent_list = [latent[1] for latent in latents]

    # convert latent_list to numpy array
    X = np.array(latent_list)
    num_clusters = buffer_size
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(X)
    
    for i in range(num_clusters):
        # print out filenames of all latents in cluster
        print([latents[j][0] for j in np.where(kmeans.labels_ == i)[0]])
        cluster = np.where(kmeans.labels_ == i)[0]
        # get the index of the latent with the smallest distance to the centroid
        centroid = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(X[cluster] - centroid, axis=1)
        selected_latent = cluster[np.argmin(distances)]
        selected_indices.append(selected_latent)

    return selected_indices

def get_buffer(celeb, folder, buffer_size):
    print(f'Getting kmeans size {buffer_size} buffer for {celeb} {folder}')
    latents = get_all_latents(celeb, folder)
    distances_map = json.load(open(os.path.join(root_dir, celeb, 'mean_distances.json')))
    selected_indices = select_indices(latents, distances_map, buffer_size)
    
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
    parser.add_argument("--device", default='0', help="device number")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    initialize_folder(args.celeb)
    get_distances(args.celeb)

    # Uncomment the following lines if you want to run get_buffer for each folder
    buffer_size = 5
    for i in range(1, 10):
        get_buffer(args.celeb, str(i), buffer_size)