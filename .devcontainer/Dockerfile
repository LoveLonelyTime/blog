FROM node:24-alpine

# Install tools
RUN apk add --no-cache \
    git \
    bash \
    openssh-client \
    curl \
    ca-certificates \
    build-base \
    python3 \
    pkgconfig

# Python
ENV PYTHON=/usr/bin/python3

# Set workspace
WORKDIR /workspace
