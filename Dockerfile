FROM python:3

ENV POETRY_VERSION=1.1.4

COPY src/ /src
COPY train.py pyproject.toml poetry.lock /
COPY --chown=55 train.sh /

RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

ENTRYPOINT python -m train.py