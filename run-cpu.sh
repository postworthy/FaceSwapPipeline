#!/bin/bash
cd "$(dirname "$0")"

docker build -f Dockerfile . -t faceswap-pipeline:latest
docker run -it -v ./faces:/app/faces/ -v ./output:/app/output/ faceswap-pipeline:latest