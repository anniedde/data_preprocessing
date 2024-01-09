import argparse
from process_celeb_videos import download_videos, celeb_video_map, combine_dataset_json, combine_test_dataset_json, process_test_subfolders
import os
import requests
import torch
import multiprocessing
import time
import traceback
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from pytube import YouTube

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

def download_video(args):
    celeb, video = args
    i = int(video.split('.')[0])
    video_urls = celeb_video_map[celeb]
    video_urls = sorted(video_urls, key=lambda x: YouTube(x).publish_date)

    vid_dir = os.path.join('videos', celeb)
    if not os.path.isdir(vid_dir):
        os.makedirs(vid_dir)
    url = video_urls[i]
    yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    print('Downloading video: ', yt.title)
    yt.streams \
        .filter(progressive=False, file_extension='mp4') \
        .order_by('resolution') \
        .desc() \
        .first() \
        .download(output_path=vid_dir,
                    filename=f'{i}.mp4')

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

parser = argparse.ArgumentParser()
parser.add_argument('--celeb', help='Specify the name of the celebrity', required=True)

args = parser.parse_args()
celeb = args.celeb

num_gpus = torch.cuda.device_count()
num_vids = len(celeb_video_map[celeb])
"""
videos = [(celeb, f"{i}.mp4") for i in range(1, 10)]

with Pool(8) as pool:
    pool.map(download_extract, videos)

processes = []
for i in range(num_gpus):
    commands = []
    for j in range(i, num_vids, num_gpus):
        command = f"python process_celeb_videos.py --celeb={celeb} --video={j} --gpu={i} -b"
        commands.append(command)
    #print('commands: ', commands)
    cmd = ' ; '.join(commands)
    process = multiprocessing.Process(target=run_command, args=(cmd, ))
    processes.append(process)
    process.start()
    
for process in processes:
    process.join()
"""

try:
    project_latent_codes_distributed(celeb)
except Exception as e:
    print(e)
    notify(f'Error in projecting latent codes of {celeb}: {e}')
    exit()

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
process_test_subfolders(celeb)


notify(f'Finished processing all videos of {celeb}!')
print('done!')
