FROM tensorflow/tensorflow:latest-gpu

ENV POETRY_VERSION=1.1.4
ENV CODE_DIRECTORY=/code
ENV PYTHONPATH "${PYTHONPATH}:${CODE_DIRECTORY}"

WORKDIR CODE_DIRECTORY

COPY mlbpestimation/ ./mlbpestimation
COPY pyproject.toml poetry.lock ./

RUN apt update
RUN apt install --yes libsndfile1 libpq-dev gcc
RUN pip install --upgrade pip
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

ENTRYPOINT python -m mlbpestimation.train