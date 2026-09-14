#!/bin/zsh
# Autonomous encoder arch/hyperparam sweep on proven data gen014_13.
# Pipeline: train (GPU) -> 30g screen vs champion gen092 (CPU) -> log.
# Survivors (screen >= 63%) are flagged; 100g vs heuristic confirm is done by the operator.
# Usage: sweep_enc.sh   (config list is inline below)
cd "$(dirname "$0")/../../.."  # -> リポジトリ直下
DATA=research/nnue/data/gen014_13.jsonl
CHAMP=gen092
SUM=research/nnue/data/enc/sweep_summary.txt
echo "=== sweep start $(date '+%H:%M:%S') data=$DATA champ=$CHAMP ===" >> "$SUM"

# config lines: "GEN SEED E LR EPOCHS"  (LR/EPOCHS override defaults 1e-3/100)
configs=(
  "096 3 32 1e-3 100"
  "097 5 32 1e-3 100"
  "098 3 24 1e-3 100"
  "099 5 24 1e-3 100"
  "100 3 16 5e-4 100"
  "101 5 16 5e-4 100"
  "102 7 32 1e-3 100"
  "103 9 24 1e-3 100"
  "104 11 16 1e-3 100"
  "105 7 16 5e-4 100"
  "106 13 32 1e-3 100"
  "107 9 32 5e-4 100"
)

for cfg in "${configs[@]}"; do
  set -- ${=cfg}
  G=$1; S=$2; E=$3; LR=$4; EP=$5
  LOG=research/nnue/data/enc/train_gen${G}.log
  echo "--- gen${G}: seed=$S E=$E lr=$LR ep=$EP start $(date '+%H:%M:%S') ---" >> "$SUM"
  python3 research/nnue/train.py --data "$DATA" --generation "$G" \
    --hidden1 96 --hidden2 24 --hidden3 24 \
    --stack-encoder-dim "$E" --batch-size 1024 --lr "$LR" --epochs "$EP" \
    --seed "$S" --device auto > "$LOG" 2>&1
  BEST=$(grep -o 'bestValLoss=[0-9.]*' "$LOG" | head -1)
  # 30-game screen vs champion (filter only)
  zsh research/nnue/bench/pmatch.sh "nnue:research/models/gen${G}.json:d64:120" "nnue:research/models/${CHAMP}.json:d64:120" \
    30 5 "research/nnue/data/enc/screen_gen${G}" 700 >/dev/null 2>&1
  RES=$(cat "research/nnue/data/enc/screen_gen${G}.agg.txt" 2>/dev/null)
  echo "gen${G} seed=$S E=$E lr=$LR $BEST | screen_vs_${CHAMP}: $RES" >> "$SUM"
done
echo "=== sweep done $(date '+%H:%M:%S') ===" >> "$SUM"
