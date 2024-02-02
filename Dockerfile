FROM mambaorg/micromamba:bookworm-slim

WORKDIR /booru-classifier

COPY conda-lock.yml ./

SHELL ["/bin/micromamba", "run", "-n", "base", "/bin/bash", "-c"]

RUN micromamba install -f conda-lock.yml -y
RUN micromamba clean --all -y

RUN rm conda-lock.yml
COPY *.py ./
COPY /data/parameters.py ./
