#!/bin/zsh
# usage: train_one_enc.sh <seed> <gen> <data> <opponentModel> <encDim>
cd /Users/akky/Documents/HTML/JULIUS
S=$1; G=$2; DATA=$3; OPP=$4; E=$5
LOG=tools/nnue/data/enc/train_gen0$G.log
python3 tools/nnue/train.py --data "$DATA" --generation "$G" --hidden1 96 --hidden2 24 --hidden3 24 \
  --stack-encoder-dim "$E" --batch-size 1024 --seed "$S" --device auto > "$LOG" 2>&1
tail -1 "$LOG"
zsh tools/nnue/bench/pmatch.sh "nnue:models/gen0$G.json:d64:120" "nnue:models/$OPP.json:d64:120" 30 5 "tools/nnue/data/enc/screen_gen0$G" 700
echo "RESULT gen0$G vs $OPP:"
cat "tools/nnue/data/enc/screen_gen0$G.agg.txt"
