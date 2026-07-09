#!/bin/zsh
# usage: train_one.sh <seed> <gen> <data> <opponentModelPath>
cd /Users/akky/Documents/HTML/JULIUS
S=$1; G=$2; DATA=$3; OPP=$4
python3 tools/nnue/train.py --data $DATA --generation $G --hidden1 96 --hidden2 24 --hidden3 24 --seed $S --device auto 2>&1 | tail -2
zsh tools/nnue/bench/pmatch.sh "nnue:models/gen0$G.json:d64:120" "nnue:models/${OPP}.json:d64:120" 30 5 tools/nnue/data/sp13/screen_gen0$G 700
echo "RESULT gen0$G vs ${OPP}: $(cat tools/nnue/data/sp13/screen_gen0$G.agg.txt)"
