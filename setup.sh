#!/bin/bash

conda create -n lungseg python=3.10 -y
conda activate lungseg

pip install -r requirements.txt

echo "Environment setup complete."

