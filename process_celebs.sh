#!/bin/bash

# set CUDA_VISIBLE_DEVICES to 2
export CUDA_VISIBLE_DEVICES=2

# List of names
names=("Priyanka")

# Path to the Python script
python_script="/playpen-nas-ssd/awang/data/distribute_process_celeb_mystyle.py"

# Loop through each name and run the Python script
for name in "${names[@]}"
do
    echo "Running Python script for name: $name"
    python $python_script --celeb $name --gpus 4,5,6,7 -b
done