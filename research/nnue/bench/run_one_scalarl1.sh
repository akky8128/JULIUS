#!/bin/zsh
# usage: run_one_scalarl1.sh <gen> <seed> <E> <lr> <epochs> <data> <champModel> <device>
set -e
cd /Users/akky/Documents/HTML/JULIUS
G=$1; S=$2; E=$3; LR=$4; EP=$5; DATA=$6; CHAMP=$7; DEV=$8
LOG=research/nnue/data/enc/train_gen${G}.log
python3 research/nnue/train.py --data "$DATA" --generation "$G" \
  --hidden1 96 --hidden2 24 --hidden3 24 \
  --stack-encoder-dim "$E" --scalar-in-l1 --batch-size 1024 --lr "$LR" --epochs "$EP" \
  --seed "$S" --device "$DEV" > "$LOG" 2>&1
BEST=$(grep -o 'bestValLoss=[0-9.]*' "$LOG" | head -1)
zsh research/nnue/bench/pmatch.sh "nnue:research/models/gen${G}.json:d64:120" "nnue:research/models/${CHAMP}.json:d64:120" \
  30 5 "research/nnue/data/enc/screen_gen${G}" 700 >/dev/null 2>&1
RES=$(cat "research/nnue/data/enc/screen_gen${G}.agg.txt" 2>/dev/null)
echo "RESULT gen${G} seed=$S E=$E lr=$LR dev=$DEV scalarInL1=1 $BEST | screen_vs_${CHAMP}: $RES"
