import os

for celeb in ['Cristiano', 'Michelle', 'Priyanka']:
    for folder in range(10):
        folder_path = f"/playpen-nas-ssd/awang/data/videos/{celeb}/{folder}"
        cropped_frames_path = os.path.join(folder_path, 'cropped_frames')
        
        if os.path.exists(cropped_frames_path):
            os.system(f"rm -rf {cropped_frames_path}")