import os
import os
import glob
import json
import shutil

# get name of the folder that this file is in
celeb = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

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
            os.remove(os.path.join(folder, f'{j}_latent.npy'))
            os.remove(os.path.join(folder, f'{j}_mirror_latent.npy'))

        #os.remove(os.path.join('all', 'train', 'preprocessed', f'{i}_{j}_latent.npy'))
        #os.remove(os.path.join('all', 'train', 'preprocessed', f'{i}_{j}_mirror_latent.npy'))
        #shutil.copy(os.path.join(folder, f'{i}_{j}_latent.npy'), os.path.join('all', 'train', 'preprocessed', f'{i}_{j}_latent.npy'))
        #shutil.copy(os.path.join(folder, f'{i}_{j}_mirror_latent.npy'), os.path.join('all', 'train', 'preprocessed', f'{i}_{j}_mirror_latent.npy'))
        #assert os.path.isfile(os.path.join(folder, f'{i}_{j}_latent.npy'))
        #assert os.path.isfile(os.path.join(folder, f'{i}_{j}_mirror_latent.npy'))

    dataset_dest_loc = os.path.join(folder, 'dataset.json')
    dataset_src_loc = os.path.join(f'/playpen-nas-ssd/awang/data/{celeb}/', str(i), 'train', 'preprocessed', 'dataset.json')

    with open(dataset_src_loc, 'r') as f:
        dataset_src = json.load(f)

    # rename the labels in the dataset
    for entry in dataset_src['labels']:
        entry[0] = f'{i}_{entry[0]}'

    with open(dataset_dest_loc, 'w') as f:
        json.dump(dataset_src, f)