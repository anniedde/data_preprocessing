import shutil
from pytube import YouTube
import os
import sys
import cv2
from PIL import Image
import numpy as np
from deepface import DeepFace
import contextlib
from random import shuffle
sys.path.append('/playpen-nas-ssd/awang/eg3d/ffhq')

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
               'https://www.youtube.com/watch?v=YX5No8y31QQ'
               ]
}

def download_videos():
    for celeb in celeb_video_map:
        video_urls = celeb_video_map[celeb]
        video_urls = sorted(video_urls, key=lambda x: YouTube(x).publish_date)

        vid_dir = os.path.join('videos', celeb)
        if not os.path.isdir(vid_dir):
            os.makedirs(vid_dir)
        for i, url in enumerate(video_urls):
            yt = YouTube(url)
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
    output_folder = os.path.join(vid_dir, video.split('.')[0], 'frames')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # Creating a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)
    
    # Loop until the end of the video
    count = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = np.array(frame)
        cv2.imwrite(f"{output_folder}/%06d.png" % (count), frame) 

        count += 1
        print(count)

    # release the video capture object
    cap.release()
    cv2.destroyAllWindows()

def extract_frames():
    for celeb in os.listdir('videos'):
        vid_dir = os.path.join('videos', celeb)
        for video in os.listdir(vid_dir):
            extract_frames_video(celeb, video)

def crop_frames_video(celeb, video):
    vid_dir = os.path.join('videos', celeb)
    frame_dir = os.path.join(vid_dir, video, 'frames')
    cropped_frames_dir = os.path.join(vid_dir, video, 'cropped_frames')
    if os.path.isdir(cropped_frames_dir):
        shutil.rmtree(cropped_frames_dir, ignore_errors=True)
    os.makedirs(cropped_frames_dir)
    for j, frame in enumerate(sorted(os.listdir(frame_dir))):
        frame_path = os.path.join(frame_dir, frame)
        img = Image.open(frame_path)
        try:
            face_objs = DeepFace.extract_faces(img_path=frame_path, 
                                                target_size=(1024, 1024),
                                                align=False,
                                                enforce_detection=False)
        except:
            face_objs = []
        for i, face_obj in enumerate(face_objs):
            face = (face_obj['face'] * 255).astype(np.uint8)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            print('confidence: ', face_obj['confidence'])
            if face_obj['confidence'] > 4:
                region = face_obj['facial_area']
                print('region:', region)
                face = img.crop((max(region['x'] - region['w'], 0),
                                    max(region['y'] - region['h'], 0), 
                                    min(region['x'] + 2 * region['w'], img.size[0] - 1), 
                                    min(region['y'] + 2 * region['h'], img.size[1] - 1)))
                face.save(os.path.join(cropped_frames_dir, f'{frame.split(".")[0]}_{i}.png'))
            #cv2.imwrite(os.path.join(cropped_frames_dir, f'{frame.split(".")[0]}_{i}.png'), face)
        print('Processed video: ', video, ', frame: ', j)

def crop_frames():
    for celeb in os.listdir('videos'):
        vid_dir = os.path.join('videos', celeb)
        print('vid_dir: ', vid_dir)
        for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
            print('video: ', video)
            frame_dir = os.path.join(vid_dir, video, 'frames')
            cropped_frames_dir = os.path.join(vid_dir, video, 'cropped_frames')
            if os.path.isdir(cropped_frames_dir):
                shutil.rmtree(cropped_frames_dir, ignore_errors=True)
            os.makedirs(cropped_frames_dir)
            for j, frame in enumerate(sorted(os.listdir(frame_dir))):
                frame_path = os.path.join(frame_dir, frame)
                img = Image.open(frame_path)
                try:
                    face_objs = DeepFace.extract_faces(img_path=frame_path, 
                                                       target_size=(1024, 1024),
                                                       align=False,
                                                       enforce_detection=False)
                except:
                    face_objs = []
                for i, face_obj in enumerate(face_objs):
                    face = (face_obj['face'] * 255).astype(np.uint8)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    print('confidence: ', face_obj['confidence'])
                    if face_obj['confidence'] > 4:
                        region = face_obj['facial_area']
                        print('region:', region)
                        face = img.crop((max(region['x'] - region['w'], 0),
                                         max(region['y'] - region['h'], 0), 
                                         min(region['x'] + 2 * region['w'], img.size[0] - 1), 
                                         min(region['y'] + 2 * region['h'], img.size[1] - 1)))
                        face.save(os.path.join(cropped_frames_dir, f'{frame.split(".")[0]}_{i}.png'))
                    #cv2.imwrite(os.path.join(cropped_frames_dir, f'{frame.split(".")[0]}_{i}.png'), face)
                print('Processed video: ', video, ', frame: ', j)

def filter_frames():
    # select frames where the face matches reference image
    for celeb in os.listdir('videos'):
        vid_dir = os.path.join('videos', celeb)
        print('vid_dir: ', vid_dir)
        for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
            print('video: ', video)
            cropped_frames_dir = os.path.join(vid_dir, video, 'cropped_frames')
            selected_frames_dir = os.path.join(vid_dir, video, 'selected_frames')
            if not os.path.isdir(selected_frames_dir):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(selected_frames_dir)
                os.makedirs(selected_frames_dir)
                for frame in os.listdir(cropped_frames_dir):
                    frame_path = os.path.join(cropped_frames_dir, frame)
                    print('frame path: ', frame_path)
                    try:
                        verify = DeepFace.verify(frame_path, os.path.join(vid_dir, 'reference.png'))
                        if verify['verified']:
                            shutil.copy(frame_path, selected_frames_dir)
                            print('verified frame: ', frame_path)
                    except ValueError:
                        continue
                
def remove_closed_eyes():
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    for celeb in os.listdir('videos'):
        vid_dir = os.path.join('videos', celeb)
        for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
            print('video: ', video)

            selected_frames_dir = os.path.join(vid_dir, video, 'selected_frames')
            eyes_open_dir = os.path.join(vid_dir, video, 'eyes_open_frames')
            if os.path.isdir(eyes_open_dir):
                shutil.rmtree(eyes_open_dir, ignore_errors=True)
            os.makedirs(eyes_open_dir)
            
            for i, frame in enumerate(sorted(os.listdir(selected_frames_dir))):
                face = cv2.imread(os.path.join(selected_frames_dir, frame))
                eyes = eye_cascade.detectMultiScale(face,scaleFactor=1.2,minNeighbors=5) 
                print(len(eyes))
                if len(eyes) > 1:
                    cv2.imwrite(os.path.join(eyes_open_dir, frame), face)
            dest_dir = os.path.join(vid_dir, video, 'final_subset')
            if not os.path.isdir(dest_dir):
                os.makedirs(dest_dir)
            eyes_open_frames = os.listdir(eyes_open_dir)
            shuffle(eyes_open_frames)
            for j, frame in enumerate(eyes_open_frames[:60]):
                shutil.copy(os.path.join(eyes_open_dir, frame), os.path.join(dest_dir, frame))

def split_train_test():
    for celeb in os.listdir('videos'):
        vid_dir = os.path.join('videos', celeb)
        if not os.path.isdir(celeb):
            os.makedirs(celeb)
        for video in sorted([v for v in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, v))]):
            print('video: ', video)
            eyes_open_frames_dir = os.path.join(vid_dir, video, 'eyes_open_frames')
            dest_dir = os.path.join(celeb, video)
            for folder in ['all', 'train', 'test']:
                if not os.path.isdir(os.path.join(dest_dir, folder)):
                    os.makedirs(os.path.join(dest_dir, folder))
            
            for i, frame in enumerate(sorted(os.listdir(eyes_open_frames_dir))):
                shutil.copy(os.path.join(eyes_open_frames_dir, frame), os.path.join(dest_dir, 'all', f'{i}.png'))

            all_frames = os.listdir(os.path.join(dest_dir, 'all'))
            shuffle(all_frames)
            train_frames = all_frames[:20]
            test_frames = all_frames[21:40]

            for j, train_frame in enumerate(train_frames):
                shutil.copy(os.path.join(dest_dir, 'all', train_frame), os.path.join(dest_dir, 'train', f'{j}.png'))

            for k, test_frame in enumerate(test_frames):
                shutil.copy(os.path.join(dest_dir, 'all', test_frame), os.path.join(dest_dir, 'test', f'{k}.png'))

def process_train_folder():

    for celeb in os.listdir('videos'):
        root = celeb
        year_folders = os.listdir(root)
        for year_folder in year_folders:
            folders = ['test', 'train']

            for folder in folders:
                indir = os.path.join(root, year_folder, folder)
                cmd = f'python /playpen-nas-ssd/awang/eg3d/dataset_preprocessing/ffhq/preprocess_in_the_wild.py  \
                    --indir={indir}'
                os.system(cmd)

if __name__ == '__main__':
    #extract_frames()
    #crop_frames()
    #filter_frames()
    remove_closed_eyes()
    #split_train_test()
    #process_train_folder()