FROM python:3.12-alpine

ARG TENSORSTORE_URL="https://files.pythonhosted.org/packages/c8/30/6adcaa7a4e17102addcf54809ead8f136d8986a197a71d0c6a4d097c4960/tensorstore-0.1.52-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
ARG PYARROW_URL="https://files.pythonhosted.org/packages/2e/92/35ca0cf2ca392172c8a269bd7b62bcc8fbcff32492c5cd9bcbbf1adf0541/pyarrow-15.0.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

WORKDIR /booru-classifier

RUN apk update && apk add --virtual curler curl

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN curl -O ${TENSORSTORE_URL}
RUN curl -O ${PYARROW_URL}

RUN apk del curler

ENV PATH="/root/.local/bin:$PATH"

COPY poetry.lock ./
COPY pyproject.toml ./

RUN apk add --virtual installer gcc build-base

RUN poetry install

RUN apk del installer

RUN apk add vim

COPY *.py ./

ENTRYPOINT ["/bin/sh"]
