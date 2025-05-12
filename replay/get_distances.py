import os, shutil, sys
import torch
import json

replay_methods = ['constrained_ransac_buffer_5_mean',
                  'constrained_ransac_buffer_5_median',
                  'constrained_ransac_buffer_10_mean',
                  'constrained_ransac_buffer_10_median']

for replay_method in replay_methods:
    harry_folder = f'/playpen-nas-ssd/awang/data/replay/{replay_method}/Harry'

    save_dict = {}
    for vid_num in range(10):
        anchors_folder = os.path.join(harry_folder, str(vid_num), 'train', 'anchors')

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
    
    with open(f'/playpen-nas-ssd/awang/data/replay/{replay_method}/Harry/mean_distances.json', 'w') as f:
        json.dump(save_dict, f)
    


