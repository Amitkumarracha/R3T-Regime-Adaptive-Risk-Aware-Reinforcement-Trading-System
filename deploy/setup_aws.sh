#!/bin/bash
# Run on fresh Ubuntu 22.04 t2.micro
echo "Starting Project Laplace Setup..."

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

# Assuming repo is already cloned or we are inside it
python3 -m venv .venv
source .venv/bin/activate

echo "Installing requirements..."
pip install -r deploy/requirements.txt

echo "Registering systemd service..."
sudo cp deploy/laplace.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable laplace

echo "Setup complete! Please configure config/.env and then run: sudo systemctl start laplace"
