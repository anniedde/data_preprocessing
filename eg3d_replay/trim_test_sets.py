import os, shutil

# get current directory
cwd = os.getcwd()

for celeb in os.listdir(cwd):
    if os.path.isdir(celeb):
        videos = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for video in videos:
            test_dir = os.path.join(cwd, celeb, video, 'test', 'preprocessed')
            for file in os.listdir(test_dir):
                if file.endswith('.png') or file.endswith('.npy'):
                    num = int(file.split('.')[0].split('_')[0])
                    if num > 9:
                        os.remove(os.path.join(test_dir, file))
                        print('removed', file)
                elif os.path.isdir(os.path.join(test_dir, file)):
                    num = int(file.split('_')[0])
                    if num > 9:
                        shutil.rmtree(os.path.join(test_dir, file))
                        print('removed', file)
