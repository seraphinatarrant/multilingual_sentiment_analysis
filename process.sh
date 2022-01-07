#!/bin/bash

langs=$1

for lang in ${langs};
do
    time python process_experiment_results.py --config /home/ec2-user/SageMaker/efs/sgt/models/config/compressed/experiment_${lang}.yaml --output /home/ec2-user/SageMaker/efs/sgt/results/compressed/${lang} --average ensemble
done
