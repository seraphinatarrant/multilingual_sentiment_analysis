#!/bin/bash

seeds="63" # 0 5 26 42 63"
lang="de"
model="/home/ec2-user/SageMaker/efs/sgt/compressed_models/de_15.0_63_2021_12_02_04_32_35/checkpoint-23445"
batch_size="24"
epochs="15"
log_file=compressed_${lang}_${epochs}_resume

for s in $seeds; do
    time python fine_tune_models.py -l $lang --batch_size $batch_size --seed $s --epochs $epochs --compressed --project_name "compressed_v1" \
    --load_model ${model}  > logs/${log_file}_${s}.out;
done