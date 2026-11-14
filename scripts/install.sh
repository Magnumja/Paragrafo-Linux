#!/bin/sh
# Compila o Parágrafo e instala para o usuário atual (sem sudo): binário, atalho e ícone.
set -eu
cd "$(dirname "$0")/.."
id=io.github.magnumja.Paragrafo
bin="$HOME/.local/bin"
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
icons="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"

cargo build --release --locked
mkdir -p "$bin" "$apps" "$icons"
install -m 755 "${CARGO_TARGET_DIR:-target}/release/paragrafo" "$bin/paragrafo"
install -m 644 "data/$id.png" "$icons/$id.png"
# O atalho aponta para o caminho completo: ~/.local/bin nem sempre está no PATH da sessão gráfica.
sed "s|^Exec=paragrafo|Exec=$bin/paragrafo|" "data/$id.desktop" > "$apps/$id.desktop"
chmod 644 "$apps/$id.desktop"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$apps" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" 2>/dev/null || true

echo "Instalado em $bin/paragrafo. Procure por \"Parágrafo\" no menu de aplicativos."
command -v ollama >/dev/null 2>&1 || echo "Para traduzir e perguntar, instale o Ollama: https://ollama.com/download"
command -v tesseract >/dev/null 2>&1 || echo "Para ler PDFs digitalizados, instale o tesseract-ocr."
