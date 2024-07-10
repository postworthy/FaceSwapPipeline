#!/bin/bash
cd "$(dirname "$0")"

DOCKER_BUILDKIT=1 docker buildx build -f Dockerfile_Base . -t faceswap-pipeline-base:latest
DOCKER_BUILDKIT=1 docker buildx build -f Dockerfile      . -t faceswap-pipeline:latest
