#!/bin/sh

python3 -m venv venv
. venv/bin/activate

sudo apt update
sudo apt install -y libgl1 -y
sudo apt install -y swig build-essential python3-dev -y

pip install -r requirements.txt

python3 train_multiprocess.py

echo "Finished training"
