FROM faceswap-pipeline-base:latest

WORKDIR /app

ADD ./network/*.py ./network/
ADD ./util.py .
ADD ./upsampler.py .
ADD ./new_faces.py .
ADD ./masks.py .
ADD ./ghost.py .
ADD ./morph.py .
ADD ./main.py .

EXPOSE 5000

CMD ["python3", "main.py"]