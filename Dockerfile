FROM nvcr.io/nvidia/tritonserver:23.04-py3 as build

RUN pip install numpy pillow opencv-python

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y