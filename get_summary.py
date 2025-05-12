import json
import numpy as np
import cv2
import os

home_dir = '/playpen-nas-ssd/awang/data/mystyle/Priyanka'

for folder in os.listdir(home_dir):
    if folder == 'all' or folder == 'up_to_5':
        continue
    if not os.path.isdir(os.path.join(home_dir, folder)):
        continue
    preprocessed = os.path.join(home_dir, folder, 'train', 'preprocessed')
    print('folder: ', folder)

    # compile all images in preprocessed into a 4 x 5 grid and save it as folder.png in home_dir
    images = []
    for img in os.listdir(preprocessed):
        img_path = os.path.join(preprocessed, img)
        images.append(cv2.imread(img_path))
        print('size of image: ', images[-1].shape)

    images = np.array(images)
    images = np.reshape(images, (4, 5, 1024, 1024, 3))
    images = np.concatenate(images, axis=2)  # concatenate along axis 2 to create rows
    images = np.concatenate(images, axis=0)  # concatenate along axis 0 to create the grid
    print('size of final image: ', images.shape)
    cv2.imwrite(os.path.join(home_dir, folder + '.jpg'), images)