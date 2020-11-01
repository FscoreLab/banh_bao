FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 AS base

RUN apt-get update
RUN apt-get install -y python3 python3-pip git wget libsm6 libxext6 libfontconfig1 libxrender1 cron cmake swig libgl1-mesa-glx
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
RUN pip3 install --upgrade pip setuptools wheel

COPY requirements/torch.txt requirements/
RUN pip3 install --no-cache-dir -r /requirements/torch.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ADD ./bao/__init__.py ./bao/__init__.py
COPY setup.py .
RUN pip3 install -e .

COPY requirements/train.txt requirements/
RUN pip3 install --no-cache-dir -r /requirements/train.txt

COPY bao/ bao/
VOLUME models/
VOLUME data/

FROM base as train

COPY train.sh train.sh
CMD "./train.sh"

FROM base as predict

ENV FILE_ORIG=""
ENV FILE_EXPERT=""
ENV FILE_MODEL=""

ENV OUTPUT_DIR=""

CMD ["python",  "bao/inference/predict.py", "--file_orig", "$FILE_ORIG", "--file_expert", "$FILE_EXPERT", "--file_model", "$FILE_MODEL", "--output_dir", "$OUTPUT_DIR"]


