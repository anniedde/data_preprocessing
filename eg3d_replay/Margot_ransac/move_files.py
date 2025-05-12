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
import torch
import numpy as np
from collections import Counter
from copy import deepcopy

def prepare_folder(celeb):

    for i in range(10, 20):
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

def get_distances():

    save_dict = {}
    for vid_num in range(20):
        anchors_folder = os.path.join(str(vid_num), 'train', 'preprocessed')

        anchors = {}
        for anchor_file in os.listdir(anchors_folder):
            if not anchor_file.endswith('latent.npy'):
                continue
            anchor = np.load(os.path.join(anchors_folder, anchor_file)).flatten()
            anchors[anchor_file] = anchor

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
    
    with open(f'mean_distances.json', 'w') as f:
        json.dump(save_dict, f)

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
        indices_combos = [indices_combo for indices_combo in indices_combos if all([Counter([latents[i][2] for i in indices_combo])[vid] >= min_bucket_size and Counter([latents[i][2] for i in indices_combo])[vid] <= max_bucket_size for vid in unique_vids])]
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

        mean_distances = {vid_num: distance_sums_by_video[vid_num] / distance_counts_by_video[vid_num] for vid_num in distance_sums_by_video}
        print('mean_distances: ', mean_distances)
        mean_distances = [distance_sums_by_video[vid_num] / distance_counts_by_video[vid_num] for vid_num in distance_sums_by_video]
        score = np.mean(mean_distances)
        print(f'Score: {score}')
        if score < min_score:
            min_score = score
            selected_indices = indices

    selected_files = [latents[i][0] for i in selected_indices]
    return selected_files

def get_buffer(folder, buffer_size):
    print('Getting buffer for folder: ', folder)
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
    
    if os.path.exists(prev_folders[1]):
        latent_files_replay = [file for file in os.listdir(prev_folders[1]) if file.endswith('latent.npy') and 'mirror' not in file]
        for i in range(len(latent_files_replay)):
            year = int(latent_files_replay[i].split('_')[0])
            file = latent_files_replay[i]
            #latent = np.load(os.path.join(prev_folders[1], file)).flatten()
            latent = np.load(os.path.join(prev_folders[1], file))[0]
            #latent = latent[-7:]
            latent = latent.flatten()
            latent_list.append(latent)
            latent_files.append(file)
            vid_nums.append(year)
    latents = list(zip(latent_files, latent_list, vid_nums))
    distances_map = json.load(open(os.path.join('mean_distances.json')))
    selected_latent_files = select_indices(latents, distances_map, buffer_size)

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
    
    for i in range(10, 20):
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
            
    get_distances()
    for i in range(10, 20):
        get_buffer(str(i), 3)