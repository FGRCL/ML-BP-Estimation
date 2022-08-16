FROM tensorflow/tensorflow:latest-gpu

ENV POETRY_VERSION=1.1.4

WORKDIR /code

COPY mlbpestimation/ ./mlbpestimation
COPY pyproject.toml poetry.lock ./
COPY --chown=55 train.sh ./

RUN apt update
RUN apt install --yes libsndfile1 python3.9
RUN update-alternatives --install /usr/bin/python3docker  python /usr/bin/python3.9 1
RUN update-alternatives --config python
RUN python -m pip install --upgrade pip
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

ENTRYPOINT cd /code && python -m mlbpestimation.train