#!/bin/bash
cd "$(dirname "$0")"

# Check if "build" argument is passed
if [[ "$1" == "build" ]]; then
    echo "Build argument detected. Running ./build.sh..."
    ./build.sh
else
    echo "No build argument detected. Skipping build step."
fi


docker run -it --shm-size=2gb --gpus all \
    -v ./docker_models_cache/.cache/:/root/.cache/ \
    -v ./docker_models_cache/.insightface/:/root/.insightface/ \
    -v ./docker_models_cache/.superres/:/root/.superres/ \
    -v ./docker_models_cache/sadtalker/:/root/sadtalker/ \
    -v ./G_latest.pth:/app/G_latest.pth \
    -v ./G_latest_mask.pth:/app/G_latest_mask.pth \
    -v ./backbone.pth:/app/backbone.pth \
    -v ./faces:/app/faces/ \
    -v ./audio:/app/audio/ \
    -v ./output:/app/output/ \
    -v /mnt/d/TrainingData/img_align_celeba/img_align_celeba/:/img_align_celeba \
    -v /mnt/d/TrainingData/sdxl_turbo_faces/:/sdxl_turbo_faces \
    -v /mnt/d/TrainingData/FromBadges/raw:/frombadges \
    faceswap-pipeline:latest

#-v ./output:/app/output/ \
#-v /mnt/d/TrainingData/sdxl_turbo_faces:/app/output/ \