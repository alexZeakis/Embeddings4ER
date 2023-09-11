#!/bin/bash

model_types=('fasttext' 'glove')
data_dir=$1
log_dir=$2
model_dir=$3
data_dirs=('dirty_amazon_itunes' 'abt_buy' 'dirty_walmart_amazon' 'dirty_dblp_acm' 'dirty_dblp_scholar')

for j in "${!model_types[@]}"; do
    for i in "${!data_dirs[@]}"; do
        python run_deepmatcher.py ${model_types[j]} "$model_dir" "$data_dir${data_dirs[i]}" "$log_dir"
    done
done
