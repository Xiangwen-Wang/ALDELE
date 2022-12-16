#!/bin/bash

DATASET=dataset_path
pssmdir_prediction=PSSM_path

# radius=1
radius=2
# radius=3

# ngram=2
ngram=3

dim=10
side=5
window=$((2*side+1))
layer_gnn=3
layer_cnn=3
layer_nn=3
layer_output=3
lr=1e-3
lr_decay=0.5
decay_interval=10
weight_decay=1e-6
iteration=100

setting=model5
python DeTool_predict_model5.py $DATASET $pssmdir_prediction $radius $ngram $dim $window $layer_gnn $layer_cnn $layer_nn $layer_output $lr $lr_decay $decay_interval $weight_decay $iteration $setting
