#!/bin/bash

pathdir=dataset path

# radius=1
radius=2
# radius=3
# ngram=2
ngram=3
dim=10
layer_gnn=3
side=5
window=$((2*side+1))
layer_cnn=3
layer_nn=3
layer_output=3
lr=1e-3
lr_decay=0.5
decay_interval=10
weight_decay=1e-6
iteration=100

# component combination:
#   model1: 2+3,
#   model2: 2+4,
#   model3: 2+3+4,
#   model4: 1+2+3,
#   model5: 1+2+3+4.
setting=5
python ALDELE_training.py $pathdir $radius $ngram $dim $layer_gnn $window $layer_cnn $layer_nn $layer_output $lr $lr_decay $decay_interval $weight_decay $iteration $setting
