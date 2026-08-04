FROM python:3.11

WORKDIR /app/cyclops
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /app/cyclops/
RUN uv sync --frozen --no-default-groups --group test --group docs

ENV PATH="/app/cyclops/.venv/bin:$PATH"
