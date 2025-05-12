import os

def remove_files_with_mirror(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'mirror' in file and 'png' in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")


# Specify the directory to start removing files from
directory = '/playpen-nas-ssd/awang/data/mystyle/Harry/all'

# Call the function to remove files and directories
remove_files_with_mirror(directory)