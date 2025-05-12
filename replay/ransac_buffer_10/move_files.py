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

root_dir = f'/playpen-nas-ssd/awang/data/replay/ransac_buffer_10'

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

def select_indices(latents, folder, buffer_size):
    selected_indices = []

    weight = 20 / buffer_size
    probabilities = [1 for _ in range(len(latents))]
    for i in range(len(latents)):
        vid_num = latents[i][2]
        if vid_num < int(folder) - 1:
            probabilities[i] = weight
    probabilities = np.array(probabilities) / np.sum(probabilities)

    indices_combos = [np.random.choice(len(latents), buffer_size, replace=False, p=probabilities) for _ in range(1000)]
    
    min_score = np.inf
    for indices in tqdm(indices_combos):
        print('indices:', [latents[i][2] for i in indices])
        chosen_latents = [latents[i] for i in indices]
        points = [latent[1] for latent in chosen_latents]
        points = np.array(points)
        model = PCA(n_components=(buffer_size - 1)).fit(points)
        proj_vertices = model.transform(points)
        hull = ConvexHull(proj_vertices)

        total_distance = 0
        for i in range(len(latents)):
            vid_num = latents[i][2]
            latent = latents[i][1]
            dist = get_distance_to_convex_hull(latent, hull, model)
            coeff = 1
            if vid_num < int(folder) - 1:
                coeff = weight
            total_distance += coeff * dist
        score = total_distance

        if score < min_score:
            min_score = score
            selected_indices = indices

    return selected_indices

def get_buffer_ransac(celeb, folder, buffer_size):
    latents = get_all_latents(celeb, folder)

    selected_indices = select_indices(latents, folder, buffer_size)

    anchors_dest_folder = os.path.join(root_dir, celeb, folder, 'replay', 'anchors')
    images_dest_folder = os.path.join(root_dir, celeb, folder, 'replay', 'preprocessed')

    if not os.path.exists(anchors_dest_folder):
        os.makedirs(anchors_dest_folder)
    if not os.path.exists(images_dest_folder):
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
    args = parser.parse_args()

    # Uncomment the following lines if you want to run get_buffer_ransac for each folder
    buffer_size = 10
    for i in range(1, 10):
        get_buffer_ransac(args.celeb, str(i), buffer_size)