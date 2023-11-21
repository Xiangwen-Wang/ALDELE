#!/bin/bash

DATASET=path to Dataset
pssmdir=path to pssm folder
energydir=path to enery term folder
# radius=0  # w/o fingerprints (i.e., atoms).
# radius=1
radius=2
# radius=3
dim=10
# ngram=2
ngram=3

python preprocess_data.py $DATASET $pssmdir $energydir $radius $ngram $dim
