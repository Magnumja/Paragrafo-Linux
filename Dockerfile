# Ambiente de desenvolvimento e verificação: o mesmo Ubuntu 24.04 usado na CI.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl pkg-config \
    libgtk-4-dev libadwaita-1-dev libpoppler-glib-dev \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-por \
    xvfb xauth dbus dbus-x11 imagemagick socat adwaita-icon-theme librsvg2-common fonts-dejavu fonts-liberation \
    && apt-get clean
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --component clippy
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
