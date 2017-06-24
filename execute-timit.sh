#!/usr/bin/env bash

./extract-data.sh -c /home/maxim/Desktop/prominator/corpus/timit/TIMIT/TEST/ -f /home/maxim/Desktop/prominator/segmentor/configs/features.conf

./extract-data.sh -c /home/maxim/Desktop/prominator/corpus/timit/TIMIT/TRAIN/ -f /home/maxim/Desktop/prominator/segmentor/configs/features.conf

python3.5 /data/timit/prepare-dataset.py -d /home/maxim/Desktop/prominator/segmentor/data/timit/TEST/ -n testset

python3.5 /data/timit/prepare-dataset.py -d /home/maxim/Desktop/prominator/segmentor/data/timit/TRAIN/ -n trainset
