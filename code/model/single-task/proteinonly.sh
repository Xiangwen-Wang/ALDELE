#!/bin/bash

DATASET=path_of_input_folder
pssmdir=slidingwindow_pssm_path
modelfoldername=output_weights

# radius=1
radius=2
# radius=3

# ngram=2
ngram=3

dim=10
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

setting=radius$radius--ngram$ngram--dim$dim--window$window--layer_cnn$layer_cnn--layer_output$layer_output--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--weight_decay$weight_decay--iteration$iteration
python weights.py $DATASET $pssmdir $modelfoldername $radius $ngram $dim $window $layer_cnn $layer_nn $layer_output $lr $lr_decay $decay_interval $weight_decay $iteration $setting
