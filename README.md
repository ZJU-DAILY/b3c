# B3C

Source code for Bi-Channel Co-clustering on HeterogeneousInformation Networks

## Instructions

* This repository contains our code (i.e., **B3C**) as well as the baselines' code

## Requirements

- Ubuntu OS
- Python >= 3.6 
- PyTorch 1.7
- A Nvidia GPU with cuda 11.2

## Data

* The datasets we used are placed in ./datasets

## Run

1. Running example on ACM

   ```shell
   python main.py --dataset ACM --input_view 0 --lr 0.005 --n_clusters 3 --ntype P
   ```


