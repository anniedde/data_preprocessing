import os

# Get the current directory
current_dir = os.getcwd()

# Iterate over the subfolders in the current directory
for folder in os.listdir(current_dir):
    # Check if the item is a directory
    if os.path.isdir(os.path.join(current_dir, folder)):
        # Check if the subfolder named 'eyes_open' exists
        eyes_open_folder = os.path.join(current_dir, folder, 'selected_frames')
        if os.path.exists(eyes_open_folder) and os.path.isdir(eyes_open_folder):
            # Count the number of PNG files in the 'eyes_open' subfolder
            png_count = len([file for file in os.listdir(eyes_open_folder) if file.endswith('.png')])
            print(f"Number of PNG files in {folder}/selected_frames: {png_count}")