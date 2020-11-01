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

COPY bao ./bao

FROM base as train

COPY data ./data
COPY models ./models

COPY train.sh train.sh
CMD ["sh", "train.sh"]

FROM base as predict

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

COPY streamlit/ streamlit/
COPY models/ models/

CMD streamlit run streamlit/evaluate.py
