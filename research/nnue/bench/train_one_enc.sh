#!/bin/zsh
# usage: train_one_enc.sh <seed> <gen> <data> <opponentModel> <encDim>
cd "$(dirname "$0")/../../.."  # -> リポジトリ直下
S=$1; G=$2; DATA=$3; OPP=$4; E=$5
LOG=research/nnue/data/enc/train_gen0$G.log
python3 research/nnue/train.py --data "$DATA" --generation "$G" --hidden1 96 --hidden2 24 --hidden3 24 \
  --stack-encoder-dim "$E" --batch-size 1024 --seed "$S" --device auto > "$LOG" 2>&1
tail -1 "$LOG"
zsh research/nnue/bench/pmatch.sh "nnue:research/models/gen0$G.json:d64:120" "nnue:research/models/$OPP.json:d64:120" 30 5 "research/nnue/data/enc/screen_gen0$G" 700
echo "RESULT gen0$G vs $OPP:"
cat "research/nnue/data/enc/screen_gen0$G.agg.txt"
