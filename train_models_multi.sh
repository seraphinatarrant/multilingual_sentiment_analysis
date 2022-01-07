#!/bin/bash

seeds="26 0"
lang="multi"
batch_size="24"
epochs="6"
log_file=compressed_scrubbed_${lang}_${epochs}


for s in $seeds; do
    time python fine_tune_models.py -l $lang --target_lang en --batch_size $batch_size --seed $s --epochs $epochs --compressed --scrub --project_name compressed_multi > logs/${log_file}_${s}.out 2>logs/${log_file}_${s}.err;
done