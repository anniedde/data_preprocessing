import numpy as np
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder') # either upper or lower
args = parser.parse_args()

loc = args.folder + '/dataset.json'
with open(loc, "r") as f:
    dataset = json.load(f)

cameras = {}
for (filename, label) in dataset['labels']:
    entry = {}
    key = filename.split('.')[0]

    entry['pose'] = np.array(label[:16]).reshape(4, 4).tolist()
    entry['intrinsics'] = np.array(label[16:]).reshape(3,3).tolist()

    cameras[key] = entry

with open(os.path.join(args.folder, 'cameras.json'), "w") as f:
    json.dump(cameras, f)