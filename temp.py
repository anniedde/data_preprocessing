import os 
import numpy as np
import shutil

root = '/playpen-nas-ssd/awang/data/Taylor'
for folder in os.listdir(root):
    pics = sorted(os.listdir(os.path.join(root, folder)))
    train = pics[0:20]
    test = pics[21:]

    for j, train_img in enumerate(train):
        original_loc = os.path.join(root, folder, train_img)
        new_folder = os.path.join(root, folder, 'train')
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)
        shutil.copy(original_loc, os.path.join(new_folder, f'{j}.jpg'))

    for i, test_img in enumerate(test):
        original_loc = os.path.join(root, folder, test_img)
        new_folder = os.path.join(root, folder, 'test')
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)
        shutil.copy(original_loc, os.path.join(new_folder, f'{i}.jpg'))

    