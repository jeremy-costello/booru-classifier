FROM python:3.12-slim

WORKDIR /booru-classifier

RUN apt-get update && apt-get install -y curl

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

RUN apt-get install -y gcc

COPY poetry.lock ./
COPY pyproject.toml ./

RUN poetry install

COPY *.py ./

ENTRYPOINT ["/bin/sh"]
