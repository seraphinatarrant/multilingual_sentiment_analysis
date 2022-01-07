#!/bin/bash

seeds="0 5 26 42 63"
lang=$1
batch_size="10"
epochs="15"
log_file=compressed_${lang}_${epochs}

for s in $seeds; do
    time python fine_tune_models.py -l $lang --batch_size $batch_size --seed $s --epochs $epochs --compressed --project_name "compressed_v1" > logs/${log_file}_${s}.out;
done