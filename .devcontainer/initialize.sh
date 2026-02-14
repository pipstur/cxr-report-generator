#!/bin/bash
set -e
sudo rm -f /etc/apt/sources.list.d/yarn.list
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get update
sudo apt-get install -y git-lfs libgl1 openssh-client

git lfs install

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
hash -r

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
else
    echo "Virtual environment already exists, skipping..."
fi

source .venv/bin/activate

uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match \
  -r training/requirements.txt

.venv/bin/pre-commit install || true

mkdir -p ~/.ssh
sudo chown -R vscode:vscode ~/.ssh
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519.pub
chmod 644 ~/.ssh/known_hosts
ssh -T git@github.com || true
