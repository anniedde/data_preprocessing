import os, sys, shutil
import numpy as np
import cv2

os.environ['DISPLAY'] = ':1'
for video in range(0, 10):
    sharp_dir = os.path.join(str(video), 'sharp_frames')
    ref_dir = os.path.join(str(video), 'ref')
    shutil.rmtree(ref_dir, ignore_errors=True)
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    
    files = sorted(os.listdir(sharp_dir))
    while len(os.listdir(ref_dir)) < 10:
        index = np.random.randint(0, len(files))
        window_name = f'Press "y" to approve or any other key to decline'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        img = cv2.imread(os.path.join(sharp_dir, files[index]))
        cv2.imshow('image', img)
        key = cv2.waitKey()
        cv2.destroyWindow('image')
        if key == ord('y'):
            shutil.copy(os.path.join(sharp_dir, files[index]), os.path.join(ref_dir, files[index]))
            done = True
            
    