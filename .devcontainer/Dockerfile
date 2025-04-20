# First stage: Build the application using Python 3.12 and uv
FROM python:3.12 AS builder

# Install uv and set environment variables
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy and install project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Set environment variables
ENV PATH="/workspace/.venv/bin:$PATH" \
    VIRTUAL_ENV="/workspace/.venv" \
    PYTHONPATH="/workspace/modules:$PYTHONPATH" \
    PYTHONUNBUFFERED=1

# Install ipykernel
RUN --mount=type=cache,target=/root/.cache/uv \
    uv add --dev ipykernel && \
    python -m ipykernel install --user
