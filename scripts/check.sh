#!/bin/sh
# Verificação completa: higiene dos arquivos versionados, lint sem avisos e todos os testes.
# Usa uma pasta temporária e nunca toca no trabalho salvo. As partes ao vivo (Ollama) e de OCR (tesseract)
# rodam quando esses programas estão disponíveis e são puladas quando não estão.
set -eu
cd "$(dirname "$0")/.."
PARAGRAFO_LIBRARY="$(mktemp -d)"
export PARAGRAFO_LIBRARY
trap 'rm -r "$PARAGRAFO_LIBRARY"' EXIT
if git rev-parse --git-dir >/dev/null 2>&1; then
    python3 scripts/audit-source.py
fi
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --nocapture
