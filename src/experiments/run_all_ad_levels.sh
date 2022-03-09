#!/bin/bash

MODEL_TYPE=$1
SCORING_METHOD=$2

for i in $(seq 1 4); do
  python run_pipeline.py --model-type $MODEL_TYPE --scoring-method $SCORING_METHOD --evaluation-type ad$i
done