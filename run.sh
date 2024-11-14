#!/bin/bash
cd "$(dirname "$0")"

./build.sh
docker run -it --shm-size=2gb --gpus all \
    -v ./docker_models_cache/.cache/:/root/.cache/ \
    -v ./docker_models_cache/.insightface/:/root/.insightface/ \
    -v ./docker_models_cache/.superres/:/root/.superres/ \
    -v ./docker_models_cache/sadtalker/:/root/sadtalker/ \
    -v ./G_latest.pth:/app/G_latest.pth \
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