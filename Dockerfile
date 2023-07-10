FROM stablecog/cuda-torch:11.7.0-2.0.0-cudnn8-devel-ubuntu22.04-v5

ADD . .
RUN apt-get update && apt-get -y install git
RUN pip3 install -r requirements.txt --no-cache-dir
RUN git config --global credential.helper store

CMD ["python3", "main.py"]