import os, sys, shutil
import numpy as np
import cv2
import argparse

def make_reference_sets(celeb):
    celeb_dir = os.path.join('/playpen-nas-ssd/awang/data/videos', celeb)
    for video in range(0, 10):
        print(f'vid {video}')
        sharp_dir = os.path.join(celeb_dir, str(video), 'sharp_frames')
        ref_dir = os.path.join(celeb_dir, str(video), 'ref')
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)
        
        files = sorted(os.listdir(sharp_dir))
        while len(os.listdir(ref_dir)) < 10:
            print('in while loop')
            index = np.random.randint(0, len(files))
            window_name = f'Press "y" to approve or any other key to decline'
            #cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            img = cv2.imread(os.path.join(sharp_dir, files[index]))
            cv2.imshow('image', img)
            key = cv2.waitKey()
            cv2.destroyWindow('image')
            if key == ord('y'):
                shutil.copy(os.path.join(sharp_dir, files[index]), os.path.join(ref_dir, files[index]))
                done = True
            elif key == ord('q'):
                return
            

if __name__ == '__main__':
    os.environ['DISPLAY'] = ':1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, help='Specify the name of the celebrity', required=True)
    args = parser.parse_args()

    print('celeb: ', args.celeb)
    make_reference_sets(args.celeb)