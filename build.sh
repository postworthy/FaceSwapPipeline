#!/bin/bash
cd "$(dirname "$0")"

DOCKER_BUILDKIT=1 docker buildx build -f Dockerfile_Base . -t localhost:5555/faceswap-pipeline-base:latest
docker push localhost:5555/faceswap-pipeline-base:latest
DOCKER_BUILDKIT=1 docker buildx build -f Dockerfile      . -t localhost:5555/faceswap-pipeline:latest
docker push localhost:5555/faceswap-pipeline:latest
