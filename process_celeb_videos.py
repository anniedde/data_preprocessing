import shutil
import traceback
from pytube import YouTube
import os
import sys
import cv2
from PIL import Image
import numpy as np
import json
import glob
from deepface import DeepFace
import contextlib
from random import shuffle
import requests
import torch
import dlib
import argparse
import math
import time
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from facenet_pytorch import MTCNN
import multiprocessing
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image
import multiprocessing
from torchvision.transforms import functional as F
import functools
import imquality.brisque as brisque


sys.path.append('/playpen-nas-ssd/awang/eg3d/ffhq')

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message
    print(message)

celeb_video_map = {
    'Margot': ['https://www.youtube.com/watch?v=PP496qqtgg8',
               'https://www.youtube.com/watch?v=Bl5630CeYFs',
               'https://www.youtube.com/watch?v=MecVr3Bz4o0',
               'https://www.youtube.com/watch?v=OmpL8fwwhEM',
               'https://www.youtube.com/watch?v=s5bI_732Cqs',
               'https://www.youtube.com/watch?v=-ebMbqkdQdg',
               'https://www.youtube.com/watch?v=aCWD2g_glhM',
               'https://www.youtube.com/watch?v=8TEBL12U7W4',
               'https://www.youtube.com/watch?v=JeUFrZtKkn8',
               'https://www.youtube.com/watch?v=YX5No8y31QQ',
               'https://youtube.com/watch?v=AC2qbj_0zhs',
                'https://youtube.com/watch?v=GmhcN130Fn8',
                'https://youtube.com/watch?v=5vTDEj_bFh8',
                'https://youtube.com/watch?v=xG_4Xtv5ANY',
                'https://youtube.com/watch?v=XA89jSn_kmY',
                'https://youtube.com/watch?v=gkpVfQBWwDA',
                'https://youtube.com/watch?v=7R1SZg0d_NU',
                'https://youtube.com/watch?v=k0a6t4oEgv8',
                'https://youtube.com/watch?v=3pNg-T9_vyg',
                'https://youtube.com/watch?v=ZGmmiuiVEnU',
                'https://youtube.com/watch?v=xdAq0Nf1KNY',
                'https://youtube.com/watch?v=J4RzAYzYOTo',
                'https://youtube.com/watch?v=QrNk6lHyDbk',
                'https://youtube.com/watch?v=-adsabO1gJo',
                'https://youtube.com/watch?v=yvDz-F5V_qY',
                'https://youtube.com/watch?v=qNZIJmEZ91g',
                'https://youtube.com/watch?v=UtGrIh4n37w',
                'https://youtube.com/watch?v=yKC7vOhr2vM',
                'https://youtube.com/watch?v=GuWr-v3TOO8',
                'https://youtube.com/watch?v=bukpOGl8syU',
               ],
    'Jennie': [
        'https://www.youtube.com/watch?v=UcXOaUThF2w&pp=ygUTYmxhY2twaW5rIGludGVydmlldw%3D%3D',
        'https://www.youtube.com/watch?v=MkcHBp4XKxs',
        'https://www.youtube.com/watch?v=JN4ukfVs49Y',
        'https://www.youtube.com/watch?v=yFQwCxsbcSE',
        'https://www.youtube.com/watch?v=Jcn2l5j6Kq8',
        'https://www.youtube.com/watch?v=ZjGiK87EaQ0',
        'https://www.youtube.com/watch?v=_qHHczHsLCo',
        'https://www.youtube.com/watch?v=BL7D2Jc70BM',
        'https://www.youtube.com/watch?v=nCHzOKvIrzg',
        'https://www.youtube.com/watch?v=VrYQ1ilnmuA'
    ],
    'IU': [
        'https://www.youtube.com/watch?v=hfXqgH2RscE',
        'https://www.youtube.com/watch?v=E1JFROb_q04',
        'https://www.youtube.com/watch?v=s26D18o-2eo',
        'https://www.youtube.com/watch?v=pbCIQf6FjGs',
        'https://www.youtube.com/watch?v=jjEOU4lE-AQ',
        'https://www.youtube.com/watch?v=eFD_V6lGvkc',
        'https://www.youtube.com/watch?v=QhQZ8xl268c',
        'https://www.youtube.com/watch?v=ZAghZAZ23x8',
        'https://www.youtube.com/watch?v=vwekEnIEN0s',
        'https://www.youtube.com/watch?v=zgzFBheLccE'
    ],
    'Harry': [
        'https://www.youtube.com/watch?v=ImIy1TLRdf8',
        'https://www.youtube.com/watch?v=6DdCPden2J0',
        'https://www.youtube.com/watch?v=WuNNyfuPUU0',
        'https://www.youtube.com/watch?v=OPFsxbc4rzY',
        'https://www.youtube.com/watch?v=IQYLPEuAClo',
        'https://www.youtube.com/watch?v=es7jXKXzW-Q',
        'https://www.youtube.com/watch?v=fhgfH0S_zgE',
        'https://www.youtube.com/watch?v=hrAC19Ytx1k',
        'https://www.youtube.com/watch?v=LZ9OjAalD6c',
        'https://www.youtube.com/watch?v=Y3GA21oUf-0'
    ],
    'Michael': [
        'https://www.youtube.com/watch?v=wV7IZ97ehGs',
        'https://www.youtube.com/watch?v=YHZzBrRDquA',
        'https://www.youtube.com/watch?v=eUycJZ5tQmA',
        'https://www.youtube.com/watch?v=sHbBPsUr5o0',
        'https://www.youtube.com/watch?v=TpvCiu2mRA8',
        'https://www.youtube.com/watch?v=yBXStZtpfN0',
        'https://www.youtube.com/watch?v=-UtV48SGO78',
        'https://www.youtube.com/watch?v=IqNMZ-wuXJ8',
        'https://www.youtube.com/watch?v=fMFxXprEpCc',
        'https://www.youtube.com/watch?v=yJ2fKJ5iUjo'
    ],
    'Sundar': [
        'https://www.youtube.com/watch?v=gEDChDOM1_U&pp=ygUYc3VuZGFyIHBpY2hhaSBhZnRlcjoyMDE4',
        'https://www.youtube.com/watch?v=X7vVP7F3-wM',
        'https://www.youtube.com/watch?v=MJs-1QxWCbI',
        'https://www.youtube.com/watch?v=ZdLRXoX3f3g',
        'https://www.youtube.com/watch?v=n2RNcPRtAiY',
        'https://www.youtube.com/watch?v=nCqEi1T0Mtk',
        'https://www.youtube.com/watch?v=igYXPKctRyw',
        'https://www.youtube.com/watch?v=7sncuRJtWQI',
        'https://www.youtube.com/watch?v=CWTm0ccfZe4',
        'https://www.youtube.com/watch?v=R8tBERK92kk'
    ],
    'Margot_up_to_5_wplus_greedy_fixedBuffer_5': ['x', 'x', 'x', 'x', 'x', 'x'],
    'Margot_up_to_5_wplus': ['x', 'x', 'x', 'x', 'x', 'x'],
    'Harry_wplus': ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
    'IU_wplus': ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
    'Margot_up_to_5_wplus_switched': ['x', 'x', 'x', 'x', 'x', 'x']
}

class FixedHeightResize:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        w, h = img.size
        aspect_ratio = float(h) / float(w)
        new_w = math.ceil(self.size / aspect_ratio)
        return F.resize(img, (self.size, new_w))

def complete_func(stream, file_path):
    print('Download complete: ', file_path)

def download_videos(celeb):
    video_urls = celeb_video_map[celeb]
    #video_urls = sorted(video_urls, key=lambda x: YouTube(x).publish_date)

    vid_dir = os.path.join('videos', celeb)
    if not os.path.isdir(vid_dir):
        os.makedirs(vid_dir)
    for i, url in enumerate(video_urls):
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True, 
                     on_complete_callback=complete_func)
        print('Downloading video: ', yt.title)
        yt.streams \
            .filter(progressive=False, file_extension='mp4') \
            .order_by('resolution') \
            .desc() \
            .first() \
            .download(output_path=vid_dir,
                        filename=f'{i}.mp4')

def extract_frames_video(celeb, video):
    vid_dir = os.path.join('videos', celeb)
    video_path = os.path.join(vid_dir, video)
    print('video_path: ', video_path)
    output_folder = os.path.join(vid_dir, video.split('.')[0], 'frames')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    else:
        return

    # Creating a VideoCapture object to read the video
    temp_cap = cv2.VideoCapture(video_path)
    length = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp_cap.release()
    interval = 1 #length // 10000 if length > 10000 else 1
    vid_start = 0
    vid_end = length

    if length > 10000:
        vid_start = length // 2 - 5000
        vid_end = length // 2 + 5000

    # Function to extract frames from a given start and end index
    def extract_frames_chunk(start, end, video_path=video_path, output_folder=output_folder, interval=interval):
        try:
            print('starting process')
            cap = cv2.VideoCapture(video_path)
            for count in tqdm(range(start, end)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                ret, frame = cap.read()
                if not ret:
                    break
                if count % interval == 0:
                    cv2.imwrite(f"{output_folder}/%06d.png" % count, frame)
            cap.release()
        except Exception as e:
            print('Error extracting frames for video: ', video, ', celeb: ', celeb, ', start: ', start, ', end: ', end)
            traceback.print_exc()
            notify(f'Error extracting frames for video: {video}, celeb: {celeb}, start: {start}, end: {end}')
            exit()


    # Determine the number of processes to use
    num_processes = 2
    chunk_size = (vid_end - vid_start) // num_processes


    # Create a list of process objects
    processes = []
    for i in range(num_processes):
        start = vid_start + i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else vid_end
        process = multiprocessing.Process(target=extract_frames_chunk, args=(start, end))
        processes.append(process)

    start_time = time.time()

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    # Release the video capture object
    
    cv2.destroyAllWindows()
    notify(f'Finished extracting frames for {celeb} video {video}. Took {execution_time_minutes} minutes')

def extract_frames(celeb):
    #for celeb in os.listdir('videos'):
    vid_dir = os.path.join('videos', celeb)
    for video in os.listdir(vid_dir):
        if video.endswith('.mp4'):
            extract_frames_video(celeb, video)

def crop_frames_video_new(celeb, video):
    vid_dir = os.path.join('videos', celeb)
    frame_dir = os.path.join(vid_dir, video, 'frames')
    cropped_frames_dir = os.path.join(vid_dir, video, 'cropped_frames')
    if os.path.isdir(cropped_frames_dir):
        shutil.rmtree(cropped_frames_dir, ignore_errors=True)
    os.makedirs(cropped_frames_dir)

    for frame in tqdm(os.listdir(frame_dir)):
        frame_path = os.path.join(frame_dir, frame)
        #print('frame path: ', frame_path)
        try:
            face_objs = DeepFace.extract_faces(img_path = frame_path, 
                    target_size = (1024, 1024), 
                    detector_backend = 'yolov8', 
                    align=False
            )
            for i, face in enumerate(face_objs):
                face_img = face['face']
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) * 255
                print('shape of face: ', face_img.shape)
                #print('face: ', face_img)
                cv2.imwrite(os.path.join(cropped_frames_dir, f'{frame[:-4]}_{i}.png'), face_img)
        except ValueError:
            continue
        except Exception as e:
            print('Error: ', e)
            notify(f'Error filtering {celeb} video {video}: {e}')
            traceback.print_exc()

def crop_frames_video(celeb, video):
    vid_dir = os.path.join('videos', celeb)
    frame_dir = os.path.join(vid_dir, video, 'frames')
    cropped_frames_dir = os.path.join(vid_dir, video, 'cropped_frames')
    if os.path.isdir(cropped_frames_dir):
        notify(f'Frames already cropped for celeb: {celeb}, video: {video}')
        return
    os.makedirs(cropped_frames_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device: ', device)
    mtcnn = MTCNN(
        image_size=1024, margin=600, min_face_size=100,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
        device=device, keep_all=True
        )
    batch_size = 40
    frames = []
    save_paths = []
    frames_list = sorted(os.listdir(frame_dir))
    start_time = time.time()
    for j, frame in enumerate(tqdm(frames_list)):
        frame_path = os.path.join(frame_dir, frame)
        image = read_image(frame_path)
        image = torch.permute(image, (1, 2, 0))
        frames.append(image)
        #image = cv2.imread(frame_path, 1)
        #image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #frames.append(Image.fromarray(image_cv))
        
        if len(frames) >= batch_size:
            save_paths = [os.path.join(cropped_frames_dir, f'{frames_list[i].split(".")[0]}.png') for i in range(j - batch_size + 1, j + 1)]
            try:
                mtcnn(frames, save_path=save_paths)
            except Exception as e:
                traceback.print_exc() 
                print(e)
            frames = []
        
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    notify(f'Finished cropping frames for {celeb} video {video}. Took {execution_time_minutes} minutes')

def crop_frames(celeb):
    vid_dir = os.path.join('videos', celeb)
    for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
        crop_frames_video(celeb, video)

def filter_frames_video(celeb, video, threshold=0.3):
    notify(f'Starting filtering frames for {celeb} video {video}')
    vid_dir = os.path.join('videos', celeb)
    eyes_open_dir = os.path.join(vid_dir, video, 'eyes_open_frames')
    selected_frames_dir = os.path.join(vid_dir, video, 'selected_frames')
    if not os.path.isdir(selected_frames_dir):
        os.makedirs(selected_frames_dir)

    lowest_distance = float('inf')
    lowest_distance_frame = None
    frames_list = []

    frames = os.listdir(eyes_open_dir)
    shuffle(frames)
    for frame in tqdm(frames):
        frame_path = os.path.join(eyes_open_dir, frame)
        try:
            reference_paths = []
            reference_paths.append(os.path.join('references', f'{celeb}.png'))
            if os.path.isdir(os.path.join(vid_dir, video, 'ref')):
                for file in os.listdir(os.path.join(vid_dir, video, 'ref')):
                    reference_paths.append(os.path.join(vid_dir, video, 'ref', file))
            for reference_path in reference_paths:
                verify = DeepFace.verify(frame_path, reference_path,
                                        model_name='ArcFace', detector_backend='yolov8', distance_metric='cosine')
                distance = verify['distance']
                #frames_list.append((frame, distance))
                if distance > 0.6:
                    break
                if distance < threshold:
                    if not os.path.isfile(os.path.join(selected_frames_dir, frame)):
                        shutil.copy(frame_path, os.path.join(selected_frames_dir, frame.replace('.png', f'_{distance:.2f}.png')))
                    break
                    
                if verify['distance'] < lowest_distance:
                    lowest_distance = verify['distance']
                    lowest_distance_frame = frame
        except ValueError:
            continue
        except Exception as e:
            print('Error: ', e)
            notify(f'Error filtering {celeb} video {video}: {e}')
            traceback.print_exc()
    """
    count = len(os.listdir(selected_frames_dir))
    frames_list = [x for x in frames_list if x[0] not in os.listdir(selected_frames_dir)]
    frames_list = sorted(frames_list, key=lambda x: x[1])
    #for frame in frames_list[:(40 - count)]:
    #    shutil.copy(os.path.join(sharp_frames_dir, frame[0]), selected_frames_dir)

    
    if len(os.listdir(selected_frames_dir)) < 40:
        print('closest frame was: ', lowest_distance_frame)
        reference_frame = lowest_distance_frame
        print('Filtering again because did not find enough frames.  Current number of frames: ', len(os.listdir(selected_frames_dir)))
        for frame in tqdm(os.listdir(sharp_frames_dir)):
            frame_path = os.path.join(sharp_frames_dir, frame)
            try:
                verify = DeepFace.verify(frame_path, os.path.join(sharp_frames_dir, reference_frame),
                                        model_name='ArcFace', detector_backend='yolov8', distance_metric='cosine')
                if verify['distance'] < threshold:
                    if not os.path.isfile(os.path.join(selected_frames_dir, frame)):
                        shutil.copy(frame_path, selected_frames_dir)
            except ValueError:
                continue
            except Exception as e:
                print('Error: ', e)
                notify(f'Error filtering {celeb} video {video}: {e}')
                traceback.print_exc()
                break
        
        frames_list = sorted(frames_list, key=lambda x: x[1])
        #for i, frame in enumerate(frames_list[:60]):
        #    shutil.copy(os.path.join(sharp_frames_dir, frame[0]), selected_frames_dir)
    """
    notify(f'Finished filtering frames for {celeb} video {video}. Found {len(os.listdir(selected_frames_dir))} frames.')

def filter_frames(celeb):
    # select frames where the face matches reference image
    vid_dir = os.path.join('videos', celeb)
    for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
        filter_frames_video(celeb, video)

def remove_closed_eyes_frame(args, reference_ratio, detector, predictor, cropped_frames_dir, eyes_open_dir):
    print('starting')
    #items, reference_ratio, detector, predictor, cropped_frames_dir, eyes_open_dir = args
    for frame in tqdm(args):
        #frame, reference_ratio, detector, predictor, cropped_frames_dir, eyes_open_dir = item
        image = cv2.imread(os.path.join(cropped_frames_dir, frame))
        face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = detector(face, 1)
        ratio = 0
        if len(dets) > 0:
            d = dets[0]
            shape = predictor(face, d)
            eye_distance = math.sqrt((shape.part(44).y - shape.part(46).y)**2 + (shape.part(44).x - shape.part(46).x)**2)
            face_distance = math.sqrt((shape.part(22).y - shape.part(8).y)**2 + (shape.part(22).x - shape.part(8).x)**2)
            ratio = eye_distance / face_distance
        if ratio >= 0.7 * reference_ratio:
            cv2.imwrite(os.path.join(eyes_open_dir, f'{frame[:-4]}_{ratio:.2f}.png'), image)
    

def remove_closed_eyes_video(celeb, video, detector, predictor, reference_ratio):

    vid_dir = os.path.join('videos', celeb)
    cropped_frames_dir = os.path.join(vid_dir, video, 'cropped_frames')
    eyes_open_dir = os.path.join(vid_dir, video, 'eyes_open_frames')
    
    if not os.path.isdir(eyes_open_dir):
        os.makedirs(eyes_open_dir)
        
        # Get the list of frames
        frames_list = sorted(os.listdir(cropped_frames_dir))
        # Determine the maximum number of processes possible
        max_processes = multiprocessing.cpu_count()
        print('max_processes: ', max_processes)
        #max_processes = 64


        # Create a multiprocessing pool with the maximum number of processes
        chunk_size = len(frames_list) // max_processes + 1
        chunks = [frames_list[i:i+chunk_size] for i in range(0, len(frames_list), chunk_size)]

        start = time.time()
        # Process each chunk of frames in parallel
        try:
            
            processes = []
            for chunk in chunks:
                p = multiprocessing.Process(target=remove_closed_eyes_frame, args=(chunk, reference_ratio, detector, predictor, cropped_frames_dir, eyes_open_dir, ))
                p.start()
                processes.append(p)

            # Join all the processes
            for p in processes:
                p.join()
            
            #pool.imap_unordered(remove_closed_eyes_frame, chunks)
            #pool.map(remove_closed_eyes_frame, chunks)
            
        except Exception as e:
            print('Error: ', e)
            traceback.print_exc()

        end = time.time()
        print('total time: ', end - start)
        #for i, frame in enumerate(tqdm(sorted(os.listdir(cropped_frames_dir)))):
        #    remove_closed_eyes_frame(frame)

        

        notify(f'Finished removing closed eyes frames for {celeb} video {video}. Took {end - start} seconds')
    else:
        notify(f'Already removed closed eyes frames for {celeb} video {video}')
    
def get_reference_ratio(celeb):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    vid_dir = os.path.join('videos', celeb)

    reference_pic = cv2.imread(os.path.join('references', f'{celeb}.png'))
    assert reference_pic is not None
    reference_face = cv2.cvtColor(reference_pic, cv2.COLOR_BGR2RGB)
    dets = detector(reference_face, 1)
    reference_ratio = 1
    for k, d in enumerate(dets):
        shape = predictor(reference_face, d)
        eye_distance = math.sqrt((shape.part(44).y - shape.part(46).y)**2 + (shape.part(44).x - shape.part(46).x)**2)
        face_distance = math.sqrt((shape.part(22).y - shape.part(8).y)**2 + (shape.part(22).x - shape.part(8).x)**2)
        reference_ratio = eye_distance / face_distance

    return reference_ratio

def remove_closed_eyes(celeb):
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    vid_dir = os.path.join('videos', celeb)

    reference_pic = cv2.imread(os.path.join('references', f'{celeb}.png'))
    reference_face = cv2.cvtColor(reference_pic, cv2.COLOR_BGR2RGB)
    dets = detector(reference_face, 1)
    reference_ratio = 1
    for k, d in enumerate(dets):
        shape = predictor(reference_face, d)
        eye_distance = math.sqrt((shape.part(44).y - shape.part(46).y)**2 + (shape.part(44).x - shape.part(46).x)**2)
        face_distance = math.sqrt((shape.part(22).y - shape.part(8).y)**2 + (shape.part(22).x - shape.part(8).x)**2)
        reference_ratio = eye_distance / face_distance

    print('reference_ratio: ', reference_ratio)

    for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
        print('video: ', video)
        remove_closed_eyes_video(celeb, video, detector, predictor, reference_ratio)

def remove_blurry_frame(frame_list, eyes_open_dir, sharp_frames_dir, threshold):
    for frame in frame_list:
        frame_path = os.path.join(eyes_open_dir, frame)
        img = cv2.imread(frame_path)
        if img.shape[0] < 400 and img.shape[1] < 400:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm > threshold:
            shutil.copy(frame_path, os.path.join(sharp_frames_dir, frame.replace('.png', f'_{fm:.2f}.png')))

        
def remove_blurry_frames_video(celeb, video, threshold=50):
    vid_dir = os.path.join('videos', celeb)
    selected_dir = os.path.join(vid_dir, video, 'selected_frames')
    sharp_frames_dir = os.path.join(vid_dir, video, 'sharp_frames')
    if not os.path.isdir(sharp_frames_dir):
        os.makedirs(sharp_frames_dir)

    frames_list = sorted(os.listdir(selected_dir))
    max_processes = multiprocessing.cpu_count() - 10

    # Create a multiprocessing pool with the maximum number of processes
    chunk_size = len(frames_list) // max_processes + 1
    chunks = [frames_list[i:i+chunk_size] for i in range(0, len(frames_list), chunk_size)]

    start = time.time()
    # Process each chunk of frames in parallel
    try:
        processes = []
        for chunk in chunks:
            p = multiprocessing.Process(target=remove_blurry_frame, args=(chunk, selected_dir, sharp_frames_dir, threshold, ))
            p.start()
            processes.append(p)

        # Join all the processes
        for p in processes:
            p.join()
        
    except Exception as e:
        print('Error: ', e)
        traceback.print_exc()

    end = time.time()
    print('total time: ', (end - start) / 60)

    notify(f'Finished removing blurry frames for {celeb} video {video}. Found {len(os.listdir(sharp_frames_dir))} frames.')

def decide_on_img(queue, save_dir, filters):
    while True:
        msg = queue.get()
        if msg == 'END':
            break

        f = msg
        try:
            img = Image.open(f)
        except:
            continue

        for filter in filters:
            if filter(img) is False:
                break
        else:
            shutil.copy2(f, save_dir)

def quality_filter(img: Image.Image, quality_threshold: float = 45):
    quality = brisque.score(img)
    return quality < quality_threshold

def filter_image_quality_video(celeb, video, threshold):
    save_dir = os.path.join('videos', celeb, video, 'quality_frames')
    if os.path.isdir(save_dir):
        #shutil.rmtree(save_dir, ignore_errors=True)
        print('Already filtered quality frames for video: ', video)
        notify(f'Already filtered quality frames for video: {video}')
        return
    os.makedirs(save_dir)

    src_dir = os.path.join('videos', celeb, video, 'selected_frames')
    filters = [functools.partial(quality_filter, quality_threshold=threshold)]

    queue = multiprocessing.Queue()
    process = [None] * 20

    for i in range(20):
        p = multiprocessing.Process(target=decide_on_img, args=(queue, save_dir, filters))
        p.daemon = True
        p.start()
        process[i] = p

    print('Putting images in queue')
    files = glob.glob(os.path.join(src_dir, '*.png'))
    print('files: ', files)
    shuffle(files)
    for f in tqdm(files):
        queue.put(f)

    for _ in process:
        queue.put('END')
    try:
        for p in process:
            print(f'Joining on {p}')
            p.join()
    except Exception as e:
        print('Error: ', e)
        traceback.print_exc()

def copy_to_final_auto_subset_video(celeb, video):
    vid_dir = os.path.join('videos', celeb)
    dest_dir = os.path.join(vid_dir, video, 'final_auto_subset')
    selected_frames_dir = os.path.join(vid_dir, video, 'selected_frames')
    
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
        selected_frames = os.listdir(selected_frames_dir)
        shuffle(selected_frames)
        for j, frame in enumerate(selected_frames):
            try:
                shutil.copy(os.path.join(selected_frames_dir, frame), os.path.join(dest_dir, frame))
            except Exception as e:
                print('Error: ', e)
                traceback.print_exc()
                break
    else:
        notify(f'Already copied frames to final auto subset folder for {celeb} video {video}. Found {len(os.listdir(dest_dir))} frames.')
    
    tensors = []

    for file in os.listdir(dest_dir):
        image = read_image(os.path.join(dest_dir, file))
        transform = transforms.Compose([
            transforms.CenterCrop(min(image.size()[1], image.size()[2])),
            transforms.Resize((1024, 1024)),
            transforms.ConvertImageDtype(dtype=torch.float),
        ])
        transformed_tensor = transform(image)
        tensors.append(transformed_tensor)

    grid = make_grid(tensors, nrow=10, padding=0)

    all_times_dest = os.path.join(vid_dir, 'all_times_final_subsets')
    if not os.path.isdir(all_times_dest):
        os.makedirs(all_times_dest)
    save_image(grid, os.path.join(all_times_dest, f'{video}.png'))

    notify(f'Finished copying frames to final auto subset folder for {celeb} video {video}')

def copy_to_final_subset(celeb):
    vid_dir = os.path.join('videos', celeb)
    for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
        copy_to_final_subset_video(celeb, video)

def split_train_test_video(celeb, video):
    vid_dir = os.path.join('videos', celeb)
    if not os.path.isdir(celeb):
        os.makedirs(celeb)
    final_subset_dir = os.path.join(vid_dir, video, 'final_subset')
    dest_dir = os.path.join('eg3d', celeb, video)
    for folder in ['all', 'train', 'test']:
        if os.path.isdir(os.path.join(dest_dir, folder)):
            print('directory: ', os.path.join(dest_dir, folder))
            print('Already split train test for video: ', video)
            notify(f'Already split train test for video: {video}')
            return
            #shutil.rmtree(os.path.join(dest_dir, folder), ignore_errors=True)
        os.makedirs(os.path.join(dest_dir, folder))
    
    final_subset_videos = os.listdir(final_subset_dir)
    shuffle(final_subset_videos)
    final_subset_videos = final_subset_videos[:40]
    for i, frame in enumerate(final_subset_videos):
        shutil.copy(os.path.join(final_subset_dir, frame), os.path.join(dest_dir, 'all', f'{i}.png'))

    all_frames = os.listdir(os.path.join(dest_dir, 'all'))
    train_frames = all_frames[:20]
    test_frames = all_frames[20:40]

    for j, train_frame in enumerate(train_frames):
        shutil.copy(os.path.join(dest_dir, 'all', train_frame), os.path.join(dest_dir, 'train', f'{j}.png'))

    for k, test_frame in enumerate(test_frames):
        shutil.copy(os.path.join(dest_dir, 'all', test_frame), os.path.join(dest_dir, 'test', f'{k}.png'))

    print('Finished splitting train test for video: ', video)

def split_train_test():
    for celeb in os.listdir('videos'):
        vid_dir = os.path.join('videos', celeb)
        if not os.path.isdir(celeb):
            os.makedirs(celeb)
        for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
            split_train_test_video(celeb, video)

def process_train_folder(celeb, video):

    root = celeb
    year_folder = video
    folders = ['train', 'test']

    for folder in folders:
        indir = os.path.join(root, year_folder, folder)
        cmd = f'python /playpen-nas-ssd/awang/eg3d/dataset_preprocessing/ffhq/preprocess_in_the_wild.py  \
            --indir=/playpen-nas-ssd/awang/data/eg3d/{indir}'
        print(cmd)
        os.system(cmd)

def subprocess_fn(img_paths, data_dir):
    os.chdir('/playpen-nas-ssd/awang/EG3D-projector/eg3d')
    for img_path in img_paths:
        c_path = img_path.replace('png', 'npy')
        if not os.path.exists(os.path.join(data_dir, img_path.replace('.png', '_latent.npy'))):
            command = "python run_projector.py --outdir=" + data_dir + \
                " --latent_space_type w  --network=/playpen-nas-ssd/awang/eg3d/eg3d/networks/ffhqrebalanced512-128.pkl --sample_mult=2 " + \
                "  --image_path " + img_path + \
                " --c_path " + c_path
            print('processing: {}'.format(img_path))
            os.system(command)

def project_latent_codes(celeb, video):
    data_dir = f"/playpen-nas-ssd/awang/data/{celeb}/{video}/train/preprocessed"

    pngFilenamesList = []
    pngFilenamesList = glob.glob(f'{data_dir}/*.png')
    pngFilenamesList = sorted([im for im in pngFilenamesList if not os.path.exists(os.path.join(data_dir, im.replace('.png', '_latent.npy'))) ])
    # split pngFilenameList into num_processes chunks
    num_processes = 2
    chunk_size = len(pngFilenamesList) // num_processes + 1
    chunks = [pngFilenamesList[i:i+chunk_size] for i in range(0, len(pngFilenamesList), chunk_size)]

    # make a new process for each chunk
    processes = []
    for chunk in chunks:
        p = multiprocessing.Process(target=subprocess_fn, args=(chunk, data_dir, ))
        p.start()
        processes.append(p)
    # join all processes
    for p in processes:
        p.join()

def extract_camera_params_celeb(celeb, video):
    root = celeb
    year_folder = video
    folders = ['test', 'train']

    for folder in folders:
        data_loc = os.path.join('eg3d', root, year_folder, folder, 'preprocessed')
        dataset_json_loc = data_loc + '/dataset.json'

        with open(dataset_json_loc) as f:
            dataset_json = json.load(f)

            labels = dataset_json['labels']

            for entry in labels:
                img_file_name = entry[0]
                params = entry[1]
                out_loc = data_loc + '/' + img_file_name[:-3] + 'npy'
                print('out_loc: ', out_loc)
                
                np.save(out_loc, params)

# copy all photos from 0/train/preprocessed, 1/train/preprocessed, etc for years 0 through 9 to a folder named all but rename each photo 0_1.png, etc
def copy_photos(celeb, video):
    year = video
    for i in range(0, 20):
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/train/preprocessed/{}.png'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/{}_{}.png'.format(celeb, year, i)
        dst_2 = '/playpen-nas-ssd/awang/data/eg3d/{}/all/all/train_{}_{}.png'.format(celeb, year, i)
        shutil.copyfile(src, dst)
        shutil.copyfile(src, dst_2)
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/train/preprocessed/{}_mirror.png'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/{}_{}_mirror.png'.format(celeb, year, i)
        dst_2 = '/playpen-nas-ssd/awang/data/eg3d/{}/all/all/train_{}_{}_mirror.png'.format(celeb, year, i)
        shutil.copyfile(src, dst)
        shutil.copyfile(src, dst_2)

        # copy the latent codes too
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/train/preprocessed/{}_latent.npy'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/{}_{}_latent.npy'.format(celeb, year, i)
        dst_2 = '/playpen-nas-ssd/awang/data/eg3d/{}/all/all/train_{}_{}_latent.npy'.format(celeb, year, i)
        shutil.copyfile(src, dst)
        #shutil.copyfile(src, dst_2)
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/train/preprocessed/{}_mirror_latent.npy'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/{}_{}_mirror_latent.npy'.format(celeb, year, i)
        dst_2 = '/playpen-nas-ssd/awang/data/eg3d/{}/all/all/train_{}_{}_mirror_latent.npy'.format(celeb, year, i)
        shutil.copyfile(src, dst)
        #shutil.copyfile(src, dst_2)

        # copy the camera params too
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/train/preprocessed/{}.npy'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/{}_{}.npy'.format(celeb, year, i)
        shutil.copyfile(src, dst)
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/train/preprocessed/{}_mirror.npy'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/{}_{}_mirror.npy'.format(celeb, year, i)
        shutil.copyfile(src, dst)


# same thing as copy_photos() but for the test folder and without latent codes
def copy_test_photos(celeb, video):
    year = video
    for i in range(0, 20):
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/test/preprocessed/{}.png'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/test/preprocessed/{}_{}.png'.format(celeb, year, i)
        dst_2 = '/playpen-nas-ssd/awang/data/eg3d/{}/all/all/test_{}_{}.png'.format(celeb, year, i)
        shutil.copyfile(src, dst)
        shutil.copyfile(src, dst_2)
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/test/preprocessed/{}_mirror.png'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/test/preprocessed/{}_{}_mirror.png'.format(celeb, year, i)
        dst_2 = '/playpen-nas-ssd/awang/data/eg3d/{}/all/all/test_{}_{}_mirror.png'.format(celeb, year, i)
        shutil.copyfile(src, dst)
        shutil.copyfile(src, dst_2)

        # copy the camera params too
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/test/preprocessed/{}.npy'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/test/preprocessed/{}_{}.npy'.format(celeb, year, i)
        shutil.copyfile(src, dst)
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/test/preprocessed/{}_mirror.npy'.format(celeb, year, i)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/test/preprocessed/{}_{}_mirror.npy'.format(celeb, year, i)

# combine the dataset.json files from each year into one new dataset.json file in all, with the updated names of the photos from the above function
# but keep the original dataset.json files in each year's folder the same
def combine_dataset_json(celeb):
    # get the dataset.json file from each year
    times = [x for x in os.listdir(os.path.join('eg3d', celeb)) if x.isdigit()]
    for time in times:
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/train/preprocessed/dataset.json'.format(celeb, time)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/{}_dataset.json'.format(celeb, time)
        shutil.copyfile(src, dst)

    # create the dataset.json file in all
    dataset = {}
    labels = []
    # update the names of the photos in the dataset.json file
    for time in times:
        for i in range(0, 20):
            # get dataset.json file from each year
            with open('/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/{}_dataset.json'.format(celeb, time), 'r') as f:
                year_dataset = json.load(f)
                year_labels = year_dataset['labels']
                for label in year_labels:
                    img_name = label[0]
                    label[0] = '{}_{}'.format(time, img_name)
                labels += year_labels
        # delete the year_dataset from the directory
        os.remove('/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/{}_dataset.json'.format(celeb, time))
    dataset['labels'] = labels

    # save the updated dataset.json file
    with open('/playpen-nas-ssd/awang/data/eg3d/{}/all/train/preprocessed/dataset.json'.format(celeb), 'w') as f:
        json.dump(dataset, f)
    

# same thing as combine_dataset_json() but for the test folder
def combine_test_dataset_json(celeb):
    # get the dataset.json file from each year
    times = [x for x in os.listdir(os.path.join('eg3d', celeb)) if x.isdigit()]
    for time in times:
        src = '/playpen-nas-ssd/awang/data/eg3d/{}/{}/test/preprocessed/dataset.json'.format(celeb, time)
        dst = '/playpen-nas-ssd/awang/data/eg3d/{}/all/test/preprocessed/{}_dataset.json'.format(celeb, time)
        shutil.copyfile(src, dst)

    # create the dataset.json file in all
    dataset = {}
    labels = []
    # update the names of the photos in the dataset.json file
    for time in times:
        for i in range(0, 20):
            # get dataset.json file from each year
            with open('/playpen-nas-ssd/awang/data/eg3d/{}/all/test/preprocessed/{}_dataset.json'.format(celeb, time), 'r') as f:
                year_dataset = json.load(f)
                year_labels = year_dataset['labels']
                for label in year_labels:
                    img_name = label[0]
                    label[0] = '{}_{}'.format(time, img_name)
                labels += year_labels
        # delete the year_dataset from the directory
        os.remove('/playpen-nas-ssd/awang/data/eg3d/{}/all/test/preprocessed/{}_dataset.json'.format(celeb, time))
    dataset['labels'] = labels

    # save the updated dataset.json file
    with open('/playpen-nas-ssd/awang/data/eg3d/{}/all/test/preprocessed/dataset.json'.format(celeb), 'w') as f:
        json.dump(dataset, f)

def convert_dataset_json(folder_path):
    loc = os.path.join(folder_path, 'dataset.json') #args.folder + '/dataset.json'
    if not os.path.exists(os.path.join(folder_path, 'cameras.json')):
        if os.path.exists(loc):
            print('Converting dataset.json to cameras.json')
            with open(loc, "r") as f:
                dataset = json.load(f)

            cameras = {}
            for (filename, label) in dataset['labels']:
                entry = {}
                key = filename.split('.')[0]

                entry['pose'] = np.array(label[:16]).reshape(4, 4).tolist()
                entry['intrinsics'] = np.array(label[16:]).reshape(3,3).tolist()

                cameras[key] = entry

            with open(os.path.join(folder_path, 'cameras.json'), "w") as f:
                json.dump(cameras, f)

def make_folders_for_images(folder):
    image_paths = os.listdir(folder)
    for fileName in image_paths:
        if fileName.endswith('.png'):
            id_name = fileName.split('.')[0]
            image_folder = os.path.join(folder, id_name)
            if not os.path.exists(image_folder):
                os.mkdir(image_folder)
                #shutil.copy(os.path.join(folder, 'crop_1024', fileName), os.path.join(image_folder, fileName))
                shutil.copy(os.path.join(folder, fileName), os.path.join(image_folder, fileName))

def process_test_subfolders(celeb, video):
    folder_path = os.path.join('eg3d', celeb, str(video))
    # get all folders that lie in folder_path recursively named test
    for root, dirs, files in os.walk(folder_path):
        if 'test' in dirs:
            test_folder = os.path.join(root, 'test', 'preprocessed')
            print('Processing folder: ' + test_folder)
            if os.path.exists(test_folder):
                convert_dataset_json(test_folder)
                make_folders_for_images(test_folder)

def clean_up_folders(celeb):
    vid_dir = os.path.join('videos', celeb)
    for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
        if os.path.isdir(os.path.join(vid_dir, video, 'frames')):
            shutil.rmtree(os.path.join(vid_dir, video, 'frames'), ignore_errors=True)

def run_funcs_before_project_latent_code(celeb, video):
    extract_frames_video(celeb, f'{video}.mp4')
    crop_frames_video(celeb, video)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    remove_closed_eyes_video(celeb, video, detector, predictor, get_reference_ratio(celeb))

    dist_thresh = 0.4
    #shutil.rmtree(os.path.join('videos', celeb, video, 'selected_frames'), ignore_errors=True)
    while not os.path.isdir(os.path.join('videos', celeb, video, 'selected_frames')) or len(os.listdir(os.path.join('videos', celeb, video, 'selected_frames'))) < 100:
        filter_frames_video(celeb, video, dist_thresh)
        dist_thresh += 0.02

    threshold = 30
    while not os.path.exists(os.path.join('videos', celeb, video, 'sharp_frames')) or len(os.listdir(os.path.join('videos', celeb, video, 'sharp_frames'))) < 60:
        print('threshold: ', threshold)
        remove_blurry_frames_video(celeb, video, threshold)
        threshold -= 10
    
    """
    quality_thresh = 45
    #shutil.rmtree(os.path.join('videos', celeb, video, 'quality_frames'), ignore_errors=True)
    while not os.path.exists(os.path.join('videos', celeb, video, 'quality_frames')) or len(os.listdir(os.path.join('videos', celeb, video, 'quality_frames'))) < 40:
        filter_image_quality_video(celeb, video, quality_thresh)
        copy_to_final_auto_subset_video(celeb, video)
        quality_thresh += 1
    
    copy_to_final_auto_subset_video(celeb, video)
    """

def run_funcs_after_project_latent_code(celeb, video):
    #split_train_test_video(celeb, video)
    #process_train_folder(celeb, video)
    #extract_camera_params_celeb(celeb, video)
    #project_latent_codes(celeb, video)
    copy_photos(celeb, video)
    copy_test_photos(celeb, video)
    process_test_subfolders(celeb, video)

if __name__ == '__main__':
    #extract_frames()
    #crop_frames()
    #filter_frames()
    #remove_closed_eyes()
    #split_train_test()
    #process_train_folder()

    #video = '2.mp4'

    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, help='Specify the name of the celebrity', required=True)
    parser.add_argument('--gpu', type=str, default='0', help='Specify the GPU index')
    parser.add_argument('--video', type=str, default='0', help='Specify which video to process')
    parser.add_argument('-b', '--before', action='store_true', help='Run functions before project_latent_code')
    parser.add_argument('-a', '--after', action='store_true', help='Run functions after project_latent_code')
    args = parser.parse_args()

    celeb = args.celeb
    video = args.video
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    try:
        
        #download_videos(celeb)
        if args.before:
            run_funcs_before_project_latent_code(celeb, video)
            #copy_to_final_auto_subset_video(celeb, video)
        if args.after:
            #process_train_folder(celeb, video)
            #extract_camera_params_celeb(celeb, video)
            run_funcs_after_project_latent_code(celeb, video)
        
        
    except Exception as e:
        notify(f'Error processing {celeb}: {e}')
        traceback.print_exc()
        exit()

    notify(f'Done processing {celeb} video {video}!')