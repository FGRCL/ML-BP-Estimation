FROM python:3-slim

ENV POETRY_VERSION=1.1.4

WORKDIR /code

COPY mlbpestimation/ ./mlbpestimation
COPY pyproject.toml poetry.lock ./
COPY --chown=55 train.sh ./

RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

ENTRYPOINT python3 /code/mlbpestimation/train.py