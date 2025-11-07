### Install

Run the following:

```sh
python3 -m venv venv
source venv/bin/activate

sudo apt update
sudo apt install -y libgl1 -y
sudo apt install -y swig build-essential python3-dev -y

pip install -r requirements.txt
```

### Run the script

For single processing:

```sh
python3 train.py
```

For multi-processing:

```sh
python3 train_multiprocess.py
```
