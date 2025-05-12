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
    ransac_num = 5000

    unique_vids = set([latent[2] for latent in latents])
    print(f'unique_vids: {unique_vids}')

    # create a dict mapping vid_num to indices corresponding to that vid_num
    vid_to_indices = {}
    for i, latent in enumerate(latents):
        vid_num = latent[2]
        if vid_num not in vid_to_indices:
            vid_to_indices[vid_num] = []
        vid_to_indices[vid_num].append(i)

    print('vid_to_indices: ', vid_to_indices)
    target_num_unique_vids = min(buffer_size, len(unique_vids))
    
    indices_combos = []
    print('Selecting indices_combos')
    if buffer_size <= len(unique_vids):
        for i in tqdm(range(ransac_num)):
            vids = random.sample(unique_vids, target_num_unique_vids)
            selected_indices = []
            for vid in vids:
                # choose a random index from vid_to_indices[vid]
                selected_indices.append(random.choice(vid_to_indices[vid]))
            indices_combos.append(selected_indices)
    else:
        # get all possible combinations of indices
        indices_combos = list(combinations(range(len(latents)), buffer_size))
        
        min_bucket_size = buffer_size // len(unique_vids)
        max_bucket_size = min_bucket_size + (buffer_size % len(unique_vids) > 0)
        print(f'min_bucket_size: {min_bucket_size}, max_bucket_size: {max_bucket_size}')
        # filter the indices_combos to only include those with between min_bucket_size and max_bucket_size indices from each vid
        new_indices_combos = []
        print('Filtering indices_combos')
        for indices_combo in tqdm(indices_combos):
            if all([Counter([latents[i][2] for i in indices_combo])[vid] >= min_bucket_size and Counter([latents[i][2] for i in indices_combo])[vid] <= max_bucket_size for vid in unique_vids]):
                new_indices_combos.append(indices_combo)
        #indices_combos = [indices_combo for indices_combo in indices_combos if all([Counter([latents[i][2] for i in indices_combo])[vid] >= min_bucket_size and Counter([latents[i][2] for i in indices_combo])[vid] <= max_bucket_size for vid in unique_vids])]
        indices_combos = new_indices_combos
        num_combinations = len(indices_combos)
        print('Number of possible combinations: ', len(indices_combos))
        
        if num_combinations > ransac_num:
            indices_combos = []
            indices_sets = []
            vids = list(unique_vids)
            while len(indices_combos) < ransac_num:
            #for _ in tqdm(range(ransac_num)):
                # choose a random ordering of the vids
                vids_left = deepcopy(vids)
                shuffle(vids_left)
                #print('vids: ', vids_left)
                i = 0
                selected_indices = []
                vid_to_indices_temp = deepcopy(vid_to_indices)
                while len(selected_indices) < buffer_size:
                    j = i % len(vids_left)
                    vid = vids_left[j]
                    vid_indices = vid_to_indices_temp[vid]
                    #print('j: ', j, 'vid: ', vid, 'vid_indices: ', vid_indices)
                    index_to_remove = random.choice(range(len(vid_indices)))
                    selected_indices.append(vid_indices.pop(index_to_remove))
                    if len(vid_indices) == 0:
                        vids_left.remove(vid)
                    else:
                        i += 1
                #indices_combos.append(selected_indices)
                if set(selected_indices) not in indices_sets:
                    indices_combos.append(selected_indices)
                    indices_sets.append(set(selected_indices))
                    #print('len(indices_combos): ', len(indices_combos))

    print(f'indices_combos head: ')
    for indices_combo in indices_combos[:20]:
        print([latents[i][0] for i in indices_combo])
        
    min_score = np.inf
    for indices in tqdm(indices_combos):
        print(f'Files: {[latents[i][0] for i in indices]}')
        chosen_latents = [latents[i] for i in indices]
        points = [latent[1] for latent in chosen_latents]
        points = np.array(points)
        model = PCA(n_components=(buffer_size - 1)).fit(points)
        proj_vertices = model.transform(points)
        hull = ConvexHull(proj_vertices)

        distance_sums_by_video = {}
        distance_counts_by_video = {}
        for i in range(len(latents)):
            if i in indices:
                continue
            vid_num = latents[i][2]
            latent = latents[i][1]
            dist = get_distance_to_convex_hull(latent, hull, model)
            
            if vid_num not in distance_sums_by_video:
                distance_sums_by_video[vid_num] = 0
                distance_counts_by_video[vid_num] = 0
            distance_sums_by_video[vid_num] += dist
            distance_counts_by_video[vid_num] += 1

        unrepresented_vids = set([v for v in unique_vids if v not in distance_sums_by_video])
        for unrepresented_vid_num in unrepresented_vids:
            # get the average distance of latents in the indices list with this vid_num from the distances_map
            files = [latents[i][0] for i in indices if latents[i][2] == unrepresented_vid_num]
            dist = np.mean([distances_map[file] for file in files])
            distance_sums_by_video[unrepresented_vid_num] = dist
            distance_counts_by_video[unrepresented_vid_num] = 1
        
        mean_distances = [distance_sums_by_video[vid_num] / distance_counts_by_video[vid_num] for vid_num in distance_sums_by_video]
        score = np.mean(mean_distances)
        print(f'Score: {score}')
        if score < min_score:
            min_score = score
            selected_indices = indices

    return selected_indices

def get_buffer_ransac(celeb, folder, buffer_size):
    print(f'Getting constrained ransac size {buffer_size} buffer for {celeb} {folder}')
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

    #initialize_folder(args.celeb)

    # Uncomment the following lines if you want to run get_buffer_ransac for each folder
    buffer_size = 10
    for i in range(7, 10):
        get_buffer_ransac(args.celeb, str(i), buffer_size)