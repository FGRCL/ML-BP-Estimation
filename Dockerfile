FROM python:3-slim

ENV POETRY_VERSION=1.1.4

WORKDIR /code

COPY mlbpestimation/ ./mlbpestimation
COPY pyproject.toml poetry.lock ./
COPY --chown=55 train.sh ./

RUN apt update && apt install --yes libsndfile1
RUN pip install --upgrade pip
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

ENTRYPOINT cd /code && python3 -m mlbpestimation.train