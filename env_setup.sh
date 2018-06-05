#!/bin/bash
sudo apt-get update
sudo apt-get install python-pip python-dev -y
pip install virtualenv
virtualenv --no-site-packages venv
source ./venv/bin/activate
pip install -r requirements.txt
source scripts/env_var.sh