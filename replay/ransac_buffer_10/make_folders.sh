#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Path to the mystyle folder
mystyle_folder="/playpen-nas-ssd/awang/data/mystyle"

# Loop through each celeb in the mystyle folder
for folder in "$mystyle_folder"/*; do
    celeb=$(basename "$folder")
    #if folder doesn't already exist in this directory
    if [ ! -d "$celeb" ]; then
        python initialize_folder.py --celeb "$celeb"
        python move_files.py --celeb "$celeb"
    fi
    
done