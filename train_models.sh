#!/bin/bash

seeds="0 5 26 42 63"
lang="ja"
batch_size="14"
epochs="30"
log_file=${lang}_${epochs}
project_name=lr_small_exp

for s in $seeds; do
    time python fine_tune_models.py -l $lang --batch_size ${batch_size} --seed $s --epochs $epochs \
    --project_name ${project_name} --mono_lr 8e-9 > logs/${log_file}_${s}.out 2>logs/${log_file}_${s}.err;
done
