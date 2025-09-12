FROM faceswap-pipeline-base:5090

WORKDIR /app

ADD ./network/*.py ./network/
ADD ./util.py .
ADD ./upsampler.py .
ADD ./new_faces.py .
ADD ./masks.py .
ADD ./ghost.py .
ADD ./morph.py .
ADD ./main.py .
ADD ./train_dreambooth_lora_sdxl.py  .
ADD ./sd_fsw_hybrid.py  .
ADD ./inswapper.py .
ADD ./pipelines_control.py .
ADD ./gpu_utils.py .

EXPOSE 5000

CMD ["python3", "main.py"]