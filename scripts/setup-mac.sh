#!/usr/bin/env bash
# Installs uv and ffmpeg if missing, creates the project environment and checks it
# Set PTV_SETUP_DRY_RUN=1 to print the commands only.
set -euo pipefail

dry="${PTV_SETUP_DRY_RUN:-}"
run() { echo "+ $*"; [ -n "$dry" ] || "$@"; }
have() { command -v "$1" >/dev/null 2>&1; }

cd "$(dirname "$0")/.."
echo "pt-vision setup in $(pwd)"

if ! have git; then
  echo "git is missing. On macOS run: xcode-select --install   (then re-run this script)"; exit 1
fi

if ! have uv; then
  echo "installing uv (Python project manager)…"
  run sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! have ffmpeg || ! have ffprobe; then
  if have brew; then
    echo "installing ffmpeg with Homebrew…"; run brew install ffmpeg
  elif have apt-get; then
    echo "installing ffmpeg with apt…"; run sudo apt-get install -y ffmpeg
  else
    echo "ffmpeg is missing and Homebrew is not installed."
    echo "Install Homebrew first:  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "then re-run this script."; exit 1
  fi
fi

echo "creating the project environment (first time: a minute or two)…"
run uv sync --extra app --extra datasets
echo
run uv run ptv models pull --mode balanced
run uv run ptv version
run uv run pre-commit install
echo
echo "Next:  uv run ptv demo        # opens the viewer on the bundled sample clip"
echo "       uv run ptv app         # then drop your own video into the window"
