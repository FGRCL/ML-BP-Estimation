FROM python:3-slim

ENV POETRY_VERSION=1.1.4

COPY src/ /code/src
COPY train.py pyproject.toml poetry.lock /code/
COPY --chown=55 train.sh /code/

RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

ENTRYPOINT python3 ./train.py