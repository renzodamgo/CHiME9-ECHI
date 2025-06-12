#!/bin/bash

conda create --name echi_recipe python=3.11
conda activate echi_recipe
conda config --add conda-forge

conda install --file requirements.txt

pip install git+https://github.com/wavlab-speech/versa.git#egg=versa-speech-audio-toolkit

# you might want to add this to your ~/.bashrc
export PYTHONPATH=$PWD/src:$PYTHONPATH

# Install NISQA
bash external/versa/tools/setup_nisqa.sh
