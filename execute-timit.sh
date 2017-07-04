#!/usr/bin/env bash

./extract-data.sh -c /home/maxim/Desktop/prominator/corpora/timit/TIMIT/TEST/ -f /home/maxim/Desktop/prominator/segmentor/configs/features.conf
./extract-data.sh -c /home/maxim/Desktop/prominator/corpora/timit/TIMIT/TRAIN/ -f /home/maxim/Desktop/prominator/segmentor/configs/features.conf

python3.5 models/prepare-dataset-parallel.py -d /home/maxim/Desktop/prominator/segmentor/data/timit/TEST/ -o testset.csv
python3.5 models/prepare-dataset-parallel.py -d /home/maxim/Desktop/prominator/segmentor/data/timit/TRAIN/ -o trainset.csv
