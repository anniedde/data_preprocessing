from pytube import YouTube
import os

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