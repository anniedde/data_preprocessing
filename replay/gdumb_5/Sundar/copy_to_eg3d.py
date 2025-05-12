import os
import shutil

times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'all']
for time in times:
    folders = ['train', 'test']

    for folder in folders:
        raw_folder = os.path.join(str(time), folder, 'raw')

        dest_folder = os.path.join('/playpen-nas-ssd/awang/data/eg3d/Sundar', str(time), folder)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        for image in os.listdir(raw_folder):
            if image.endswith('.png'):
                shutil.copy(os.path.join(raw_folder, image), os.path.join(dest_folder, image))
                print('Copied', image)