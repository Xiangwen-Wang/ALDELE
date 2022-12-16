#!/bin/bash

DATASET=/Users/xw0201/Desktop/postdoc/NNmodel_dataset/CALB/calb_a_bsla/con_results
pssmdir=/Users/xw0201/Desktop/postdoc/NNmodel_dataset/CALB/calb_a_bsla/con_results/pssm_slidingwindow

# radius=0  # w/o fingerprints (i.e., atoms).
# radius=1
radius=2
# radius=3
dim=10
# ngram=2
ngram=3

python preprocess_data.py $DATASET $pssmdir $radius $ngram $dim
