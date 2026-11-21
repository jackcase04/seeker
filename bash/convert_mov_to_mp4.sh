#!/bin/bash

INP_DIR_NAME="video/mov/"
OUT_DIR_NAME="video/mp4/"

mkdir -p $INP_DIR_NAME
mkdir -p $OUT_DIR_NAME

for filename in $INP_DIR_NAME*
do
    FULL_PATH=$filename
    FILE_NAME=$(basename ${filename} | cut -d '.' -f 1)

    if [ ! -f "${OUT_DIR_NAME}${FILE_NAME}.mp4" ]; then
        ffmpeg -i $FULL_PATH -c:v libx264 -c:a aac "${OUT_DIR_NAME}${FILE_NAME}.mp4"
    else
        echo "${FILE_NAME}.mp4 already exists, skipping"
    fi
done