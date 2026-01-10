#!/bin/bash
set -e
sudo chown -R vscode:vscode /workspaces/cxr-report-generator/
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get update
sudo apt-get install -y git-lfs libgl1
sudo apt install openssh-client
git lfs install
python -m venv venv
source venv/bin/activate
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
pip install --upgrade pip
# pre-commit install # optionally, if you plan to push code
uv pip install -r training/requirements.txt

sudo chmod 700 ~/.ssh && sudo chmod 600 ~/.ssh/*
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts
