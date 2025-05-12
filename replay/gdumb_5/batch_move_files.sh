#!/bin/bash

# List of arguments
arguments=("Harry" "IU" "Michael" "Sundar" "Margot")

# Loop through the arguments
for arg in "${arguments[@]}"
do
    # Run move_files with the current argument
    python move_files.py --celeb "$arg" --device 1
done
