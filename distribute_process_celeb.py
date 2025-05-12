import argparse
from process_celeb_videos import download_videos, celeb_video_map, combine_dataset_json, combine_test_dataset_json, process_test_subfolders, \
    split_train_test_video, process_train_folder, extract_camera_params_celeb
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image
import os
import requests
import torch
import multiprocessing
from random import shuffle
import time
import traceback
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from pytubefix import YouTube
import shutil
import glob
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

def extract_frames(args):
    celeb, video = args
    vid_dir = os.path.join('videos', celeb)
    video_path = os.path.join(vid_dir, video)
    print('video_path: ', video_path)
    output_folder = os.path.join(vid_dir, video.split('.')[0], 'frames')
    if not os.path.isdir(output_folder):
            
        os.makedirs(output_folder)

        # Creating a VideoCapture object to read the video
        temp_cap = cv2.VideoCapture(video_path)
        length = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap.release()
        print('length: ', length)
            
        print('starting process')
        cap = cv2.VideoCapture(video_path)
        if length > 12000:
            frame_pos = (length - 12000) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        start_time = time.time()
        count = 0
        try:
            for i in tqdm(range(12000)):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(f"{output_folder}/%06d.png" % count, frame)
                count += 1
            cap.release()
        except Exception as e:
            print('Error extracting frames for video: ', video, ', celeb: ', celeb)
            traceback.print_exc()
            notify(f'Error extracting frames for video: {video}, celeb: {celeb}')
            exit()

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60

        # Release the video capture object
        
        cv2.destroyAllWindows()
        notify(f'Finished extracting frames for {celeb} video {video}. Took {execution_time_minutes} minutes')
    else:
        notify(f'Frames already extracted for celeb: {celeb}, video: {video}')

def download_video(args):
    celeb, video = args
    i = int(video.split('.')[0])
    video_urls = celeb_video_map[celeb]
    #video_urls = sorted(video_urls, key=lambda x: YouTube(x).publish_date)

    vid_dir = os.path.join('videos', celeb)
    if not os.path.isdir(vid_dir):
        os.makedirs(vid_dir)
    if not os.path.isfile(os.path.join(vid_dir, f'{i}.mp4')):
        url = video_urls[i]
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        print('Downloading video: ', yt.title)
        custom_filter = lambda stream: stream.video_codec.startswith('avc')
        stream = yt.streams \
            .filter(progressive=False, file_extension='mp4', custom_filter_functions=[custom_filter], only_video=True) \
            .order_by('resolution') \
            .desc() \
            .first()
        assert stream is not None, 'No stream found'
        stream.download(output_path=vid_dir,
                        filename=f'{i}.mp4')

def manual_filter(celeb, start, end):
    notify(f'Starting manual filter for final subsets for {celeb}')
    os.environ['DISPLAY'] = ':1'
    for video in range(start, end):
        try:
            final_auto_subset_dir = os.path.join('videos', celeb, str(video), 'final_auto_subset')
            final_auto_subset_dir = os.path.join('videos', celeb, str(video), 'eyes_open_frames')            
            final_subset_dir = os.path.join('videos', celeb, str(video), 'final_subset')
            if not os.path.isdir(final_subset_dir):
                os.makedirs(final_subset_dir)
            if len(os.listdir(final_subset_dir)) >= 30:
                print('Already have enough images for video ', video)
                continue
            print(f'Manually filtering for video {video}. Already have {len(os.listdir(final_subset_dir))} images')
            
            files = os.listdir(final_auto_subset_dir)
            shuffle(files)
            for file in files:
                if file in os.listdir(final_subset_dir):
                    continue
                window_name = f'Press "y" to approve or any other key to decline'
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                img = cv2.imread(os.path.join(final_auto_subset_dir, file))
                cv2.imshow('image', img)
                key = cv2.waitKey()
                cv2.destroyWindow('image')
                if key == ord('y'):
                    shutil.copy(os.path.join(final_auto_subset_dir, file), os.path.join(final_subset_dir, file))
                elif key == ord('q'):
                    return
                if len(os.listdir(final_subset_dir)) >= 30:
                    break
            if len(os.listdir(final_subset_dir)) < 30:
                print('Not enough images, need to manually select more images')
            tensors = []

            for file in os.listdir(final_subset_dir):
                image = read_image(os.path.join(final_subset_dir, file))
                transform = transforms.Compose([
                    transforms.CenterCrop(min(image.size()[1], image.size()[2])),
                    transforms.Resize((1024, 1024)),
                    transforms.ConvertImageDtype(dtype=torch.float),
                ])
                transformed_tensor = transform(image)
                tensors.append(transformed_tensor)

            grid = make_grid(tensors, nrow=10, padding=0)

            all_times_dest = os.path.join('videos', celeb, str(video), 'all_times_final_subsets')
            if not os.path.isdir(all_times_dest):
                os.makedirs(all_times_dest)
            save_image(grid, os.path.join(all_times_dest, f'{video}.png'))
        except Exception as e:
            print('Error in manual filter')
            continue

    notify(f'Done with manual filter for final subsets for {celeb}')

def format_folders(celeb, start, end):
    for video in range(start, end):
        split_train_test_video(celeb, str(video))
        process_train_folder(celeb, str(video))
        extract_camera_params_celeb(celeb, str(video))

def download_extract(args):
    download_video(args)
    extract_frames(args)

def run_command(cmd):
    print(cmd)
    os.system(cmd)

def project_latent_codes_distributed(celeb):
    os.chdir('/playpen-nas-ssd/awang/EG3D-projector/eg3d')
    command = f'python distribute_project_all.py --celeb={celeb}'
    os.system(command)
    os.chdir('/playpen-nas-ssd/awang/data')

def create_all_folders(celeb):
    # make the all folder if it doesn't exist
    if not os.path.exists(f'/playpen-nas-ssd/awang/data/eg3d/{celeb}/all/test/preprocessed'):
        os.makedirs(f'/playpen-nas-ssd/awang/data/eg3d/{celeb}/all/test/preprocessed')
    if not os.path.exists(f'/playpen-nas-ssd/awang/data/eg3d/{celeb}/all/all'):
        os.makedirs(f'/playpen-nas-ssd/awang/data/eg3d/{celeb}/all/all')
    # make the all folder if it doesn't exist
    if not os.path.exists(f'/playpen-nas-ssd/awang/data/eg3d/{celeb}/all/train/preprocessed'):
        os.makedirs(f'/playpen-nas-ssd/awang/data/eg3d/{celeb}/all/train/preprocessed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', help='Specify the name of the celebrity', required=True)
    parser.add_argument('-b', action='store_true', help='Run code before projecting latent codes')
    parser.add_argument('-p', action='store_true', help='Run projection code')
    parser.add_argument('-a', action='store_true', help='Run code after projecting latent codes')
    parser.add_argument('--range', help='Specify the range of videos to process', required=False)

    args = parser.parse_args()
    celeb = args.celeb
    
    num_gpus = torch.cuda.device_count()
    num_vids = len(celeb_video_map[celeb])
    print('num_vids: ', num_vids)

    if args.range:
        start, end = map(int, args.range.split(','))
    else:
        start, end = 0, 10
    videos = [(celeb, f"{i}.mp4") for i in range(start, end)]
    
    if args.b:
        with Pool(num_gpus) as pool:
            pool.map(download_extract, videos)

        processes = []
        for i in range(num_gpus):
            commands = []
            for j in range(start + i, end, num_gpus):
                command = f"python process_celeb_videos.py --celeb={celeb} --video={j} --gpu={i} -b"
                commands.append(command)
            #print('commands: ', commands)
            cmd = ' ; '.join(commands)
            process = multiprocessing.Process(target=run_command, args=(cmd, ))
            processes.append(process)
            process.start()
            
        for process in processes:
            process.join()
        
        manual_filter(celeb, start, end)

    if args.p:
        try:
            format_folders(celeb, start, end)
            create_all_folders(celeb)
            project_latent_codes_distributed(celeb)
        except Exception as e:
            print(e)
            notify(f'Error in projecting latent codes of {celeb}: {e}')
            exit()

    if args.a:
        try:
            processes = []
            for i in range(num_gpus):
                commands = []
                for j in range(i, num_vids, num_gpus):
                    command = f"python process_celeb_videos.py --celeb={celeb} --video={j} --gpu={i} -a"
                    commands.append(command)
                #print('commands: ', commands)
                cmd = ' ; '.join(commands)
                process = multiprocessing.Process(target=run_command, args=(cmd, ))
                processes.append(process)
                process.start()
                
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            for process in processes:
                process.terminate()
            print('terminated all processes')
            exit()

    
        combine_dataset_json(celeb)
        combine_test_dataset_json(celeb)
        #process_test_subfolders(celeb)


    notify(f'Finished processing all videos of {celeb}!')
