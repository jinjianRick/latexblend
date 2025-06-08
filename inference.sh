#!/bin/bash

JSON_PATH=$1                             # path to the config file

CKPT_PATH="stabilityai/stable-diffusion-xl-base-1.0" 
DEVICE="cuda:1"
PYTHON_SCRIPT="src/diffusers_sample.py"
OUTPUT_PREFIX='output/'                                                 # output path


# Extract the filename without the extension
savename=$(basename -- "$JSON_PATH" .json)

  # Run the Python command for each JSON file and set the output filename
python $PYTHON_SCRIPT \
    --ckpt "$CKPT_PATH" \
    --sdxl \
    --concept_list "$JSON_PATH" \
    --deep_replace \
    --device "$DEVICE" \
    --output_file "$OUTPUT_PREFIX$savename" \
    --seed 42
