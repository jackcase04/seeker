#!/bin/bash

INP_DIR_NAME="video/mp4/"
OUT_DIR_NAME="data/workspace/pose/test/"

for filename in $INP_DIR_NAME*
do
    FULL_PATH=$filename
    FILE_NAME=$(basename ${filename} | cut -d '.' -f 1)
    FILE_NAME_EXTENSION=$(basename ${filename})

    if [ ! -f "${OUT_DIR_NAME}${FILE_NAME}_alphapose_tracked_person.json" ]; then
        python src/extract_poses.py "${INP_DIR_NAME}${FILE_NAME_EXTENSION}" "${OUT_DIR_NAME}"
    else
        echo "Output JSON for ${FILE_NAME} already exists, skipping"
    fi
done

