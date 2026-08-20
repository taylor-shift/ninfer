# syntax=docker/dockerfile:1
#
# RunPod Serverless (load balancing) worker for NInfer.
#
# Build from the repository root:
#     docker build -f deploy/runpod/Dockerfile -t <user>/ninfer-runpod:latest .
#
# There is no handler process and no Python in the request path. ninfer-serve is
# itself the worker's HTTP server: the load balancer routes requests straight to
# it and polls its /health route. The entrypoint only resolves the cached
# artifact and sizes the context before exec'ing the engine.
#
# The image carries no model. The .ninfer artifact is supplied at run time by the
# endpoint's cached model, so one image serves every registered artifact and stays
# small enough to pull quickly onto a cold host.

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# sm_120a is the only architecture this engine accepts, and the base image's nvcc
# is the sole toolchain present, so no CUDACXX override is needed here.
RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer-serve

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY deploy/runpod/entrypoint.sh /usr/local/bin/ninfer-entrypoint

RUN chmod +x /usr/local/bin/ninfer-entrypoint

# Defaults match the measured RTX 5090 configuration for one 27B NVFP4 artifact.
# Every value is overridable from the endpoint's environment variables, and
# NINFER_MAX_CONTEXT is deliberately unset so the entrypoint sizes it from the
# GPU the worker actually lands on.
ENV NINFER_MODEL="neroued/Qwen3.8-27B-nvfp4-NInfer" \
    NINFER_KV_DTYPE=int8 \
    NINFER_KV_CAPACITY=auto \
    NINFER_PROFILE=auto \
    NINFER_SPEC=mtp \
    NINFER_DRAFT_TOKENS=3 \
    NINFER_LM_HEAD_DRAFT=1 \
    NINFER_PRESERVE_THINKING=1 \
    NINFER_PENDING_TIMEOUT_MS=120000 \
    PORT=80 \
    HEALTH_CHECK_PATH=/health

WORKDIR /workspace
EXPOSE 80
STOPSIGNAL SIGTERM

ENTRYPOINT ["/usr/local/bin/ninfer-entrypoint"]
