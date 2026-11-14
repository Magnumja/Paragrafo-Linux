#!/bin/sh
# Instala o Parágrafo com um comando só: dependências do sistema, Rust (se faltar), compilação,
# binário, atalho no menu e ícone.
#
#   ./scripts/install.sh            instala o app
#   ./scripts/install.sh --ollama   também instala o Ollama (se faltar) e baixa os modelos
#
# Só a instalação de pacotes do sistema pede sudo; o app em si vai para a sua pasta pessoal.
set -eu
cd "$(dirname "$0")/.."

with_ollama=false
for argument in "$@"; do
    case "$argument" in
        --ollama) with_ollama=true ;;
        -h | --help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Opção desconhecida: $argument" >&2; exit 2 ;;
    esac
done

step() { printf '\n==> %s\n' "$1"; }
as_root() { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi; }

step "Dependências do sistema"
if command -v apt-get >/dev/null 2>&1; then
    as_root apt-get update
    as_root apt-get install -y build-essential pkg-config curl ca-certificates \
        libgtk-4-dev libadwaita-1-dev libpoppler-glib-dev \
        tesseract-ocr tesseract-ocr-por tesseract-ocr-eng
elif command -v dnf >/dev/null 2>&1; then
    as_root dnf install -y gcc pkgconf-pkg-config curl \
        gtk4-devel libadwaita-devel poppler-glib-devel \
        tesseract tesseract-langpack-por tesseract-langpack-eng
elif command -v pacman >/dev/null 2>&1; then
    as_root pacman -S --needed --noconfirm base-devel pkgconf curl \
        gtk4 libadwaita poppler-glib \
        tesseract tesseract-data-por tesseract-data-eng
elif command -v zypper >/dev/null 2>&1; then
    as_root zypper install -y gcc pkg-config curl \
        gtk4-devel libadwaita-devel libpoppler-glib-devel \
        tesseract-ocr tesseract-ocr-traineddata-por tesseract-ocr-traineddata-eng
else
    echo "Gerenciador de pacotes não reconhecido. Instale GTK 4.12+, libadwaita 1.5+, Poppler (glib) e, se quiser OCR, o Tesseract; depois rode este script de novo." >&2
    pkg-config --exists gtk4 libadwaita-1 poppler-glib || exit 1
fi
if ! pkg-config --atleast-version=4.12 gtk4 || ! pkg-config --atleast-version=1.5 libadwaita-1; then
    echo "Esta distribuição traz GTK $(pkg-config --modversion gtk4) e libadwaita $(pkg-config --modversion libadwaita-1); o Parágrafo precisa de GTK 4.12+ e libadwaita 1.5+ (Ubuntu 24.04, Fedora 40, Debian 13 ou mais novos)." >&2
    exit 1
fi

step "Rust"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
# O Rust das distribuições costuma ser antigo demais; o rustup traz o atual para a sua pasta pessoal.
needed=1.88
current=$(cargo --version 2>/dev/null | awk '{print $2}' || true)
if [ -z "$current" ] || [ "$(printf '%s\n%s\n' "$needed" "$current" | sort -V | head -n1)" != "$needed" ]; then
    if command -v rustup >/dev/null 2>&1; then
        rustup update stable
        rustup default stable
    else
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
        . "$HOME/.cargo/env"
    fi
fi
cargo --version

step "Compilando (alguns minutos na primeira vez)"
cargo build --release --locked

step "Instalando para o seu usuário"
id=io.github.magnumja.Paragrafo
data="${XDG_DATA_HOME:-$HOME/.local/share}"
bin="$HOME/.local/bin"
apps="$data/applications"
icons="$data/icons/hicolor/256x256/apps"
mkdir -p "$bin" "$apps" "$icons"
install -m 755 "${CARGO_TARGET_DIR:-target}/release/paragrafo" "$bin/paragrafo"
install -m 644 "data/$id.png" "$icons/$id.png"
# O atalho aponta para o caminho completo: ~/.local/bin nem sempre está no PATH da sessão gráfica.
sed "s|^Exec=paragrafo|Exec=$bin/paragrafo|" "data/$id.desktop" > "$apps/$id.desktop"
chmod 644 "$apps/$id.desktop"
if command -v update-desktop-database >/dev/null 2>&1; then update-desktop-database "$apps" || true; fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then gtk-update-icon-cache -q -t "$data/icons/hicolor" 2>/dev/null || true; fi

if $with_ollama; then
    step "Ollama e modelos"
    if ! command -v ollama >/dev/null 2>&1; then
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    # Os downloads precisam do servidor; se ele não estiver no ar, sobe um só para isso.
    started=""
    if ! curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
        ollama serve >/dev/null 2>&1 &
        started=$!
        sleep 3
    fi
    ollama pull llama3.2:3b
    ollama pull bge-m3
    if [ -n "$started" ]; then kill "$started" 2>/dev/null || true; fi
fi

step "Pronto"
echo "Parágrafo instalado em $bin/paragrafo. Procure por \"Parágrafo\" no menu de aplicativos."
if ! command -v ollama >/dev/null 2>&1; then
    echo "Para traduzir e perguntar falta o Ollama: rode ./scripts/install.sh --ollama (ou instale por https://ollama.com/download)."
fi
