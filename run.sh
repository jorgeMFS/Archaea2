#!/bin/bash

pip3 install -r requirements.txt 

cd scripts || exit
chmod +x ./*.sh
bash install_tools.sh
bash prepare_and_classify_dataset.sh
