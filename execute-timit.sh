#!/usr/bin/env bash

./extract-corpus.sh -c /home/maxim/Desktop/prominator/corpora/timit/TIMIT/TEST/ -f /home/maxim/Desktop/prominator/segmentor/configs/features.conf
./extract-corpus.sh -c /home/maxim/Desktop/prominator/corpora/timit/TIMIT/TRAIN/ -f /home/maxim/Desktop/prominator/segmentor/configs/features.conf

python3.5 prepare-datasets.py -d /home/maxim/Desktop/prominator/segmentor/data/timit/TEST/ -o testset.csv -c 6
python3.5 prepare-datasets.py -d /home/maxim/Desktop/prominator/segmentor/data/timit/TRAIN/ -o trainset.csv -c 6

python3.5 train-model.py -t datasets/trainset.csv -s datasets/testset.csv -v 0.2 -c 6
