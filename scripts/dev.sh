#!/bin/sh
# Roda um comando no ambiente Linux de desenvolvimento (Ubuntu 24.04, o mesmo da CI).
# Útil para compilar e verificar a partir de outro sistema:
#   scripts/dev.sh cargo test --workspace
#   scripts/dev.sh scripts/check.sh
set -eu
cd "$(dirname "$0")/.."
docker image inspect paragrafo-linux-dev >/dev/null 2>&1 || docker build -t paragrafo-linux-dev .
exec docker run --rm \
    -v "$PWD":/src \
    -v paragrafo-cargo:/root/.cargo/registry \
    -v paragrafo-target:/target \
    -e CARGO_TARGET_DIR=/target \
    paragrafo-linux-dev "$@"
