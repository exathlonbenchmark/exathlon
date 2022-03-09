#!/bin/bash

MODEL_TYPE=$1
MODELING_DATA_SEED=$2
MODELING_SPLIT_SEED=$3
MAX_PERIODS=52

if [ $MODEL_TYPE == rnn ]; then
  SCORING_METHOD=re
elif [ $MODEL_TYPE == ae ]; then
  SCORING_METHOD=mse
else
  SCORING_METHOD=mse.ft
fi
for i in $(seq 1 $MAX_PERIODS); do
  python ../modeling/train_model.py --modeling-n-periods $i --modeling-data-seed $MODELING_DATA_SEED \
  --modeling-split-seed $MODELING_SPLIT_SEED --model-type $MODEL_TYPE
  python ../scoring/train_scorer.py --modeling-n-periods $i --modeling-data-seed $MODELING_DATA_SEED \
  --modeling-split-seed $MODELING_SPLIT_SEED --model-type $MODEL_TYPE --scoring-method $SCORING_METHOD
  python ../detection/train_detector.py --modeling-n-periods $i --modeling-data-seed $MODELING_DATA_SEED \
  --modeling-split-seed $MODELING_SPLIT_SEED --model-type $MODEL_TYPE --scoring-method $SCORING_METHOD
done
