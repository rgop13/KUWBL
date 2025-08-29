FROM nvcr.io/nvidia/pytorch:23.03-py3

USER root
# HF + Pillow 10+로 맞춤
RUN pip install --no-cache-dir \
    datasets==2.20.0 \
    pillow==10.4.0 \
    pyarrow==15.0.2 \
    numpy \
    tensorboardX==2.6.2 \
    tqdm fsspec aiohttp xxhash multiprocess

RUN mkdir -p /opt/mnist/src
WORKDIR /opt/mnist/src

USER 500:500
COPY mnist_distributed.py /opt/mnist/src/mnist_distributed.py
