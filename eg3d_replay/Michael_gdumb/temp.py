import os, shutil

videos = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'all']
for video in videos:
    for split in ['train', 'test']:
        src_dir = os.path.join(video, split, 'raw')
        dst_dir = os.path.join(video, split)

        for file in os.listdir(dst_dir):
            if os.path.isdir(os.path.join(dst_dir, file)):
                shutil.rmtree(os.path.join(dst_dir, file))