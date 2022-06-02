FROM python:3.9.7


WORKDIR /app/cyclops
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8


RUN apt-get update \
    && apt-get install -y git software-properties-common \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip
RUN pip install poetry


COPY * /app/cyclops/
RUN poetry config virtualenvs.create false \
   && poetry install --no-interaction --no-ansi
