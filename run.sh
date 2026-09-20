#!/bin/bash
# GRIP on every dataset x density of the main table (validated hyperparameters from scr/hyperparams.py, repeat 10)
gpu=${1:-0}
data_dir=${2:-./data/}
repeat=${3:-10}

for r in 0.013 0.026 0.052; do
    python main.py --gpu $gpu --raw_data_dir $data_dir --repeat $repeat --dataset_name cora --ratio $r
done
for r in 0.009 0.018 0.036; do
    python main.py --gpu $gpu --raw_data_dir $data_dir --repeat $repeat --dataset_name citeseer --ratio $r
done
for r in 0.0005 0.0025 0.005; do
    python main.py --gpu $gpu --raw_data_dir $data_dir --repeat $repeat --dataset_name arxiv --ratio $r
done
for r in 0.001 0.005 0.01; do
    python main.py --gpu $gpu --raw_data_dir $data_dir --repeat $repeat --dataset_name flickr --ratio $r
done
for r in 0.0005 0.001 0.002; do
    python main.py --gpu $gpu --raw_data_dir $data_dir --repeat $repeat --dataset_name reddit --ratio $r
done
