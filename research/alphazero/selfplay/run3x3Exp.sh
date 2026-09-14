#!/usr/bin/env bash
# run3x3Exp.sh — 3x3盤 先手有利検証の1実験を scratch から回す。
#
# 使い方:
#   EXP=exp1 UL_MAX_WHITE=4 UL_MAX_BLACK=4 UL_ELIM_UNLOCK=both  ./run3x3Exp.sh
# 環境変数（ルール）:
#   UL_MAX_WHITE / UL_MAX_BLACK / UL_ELIM_UNLOCK は gameLogic.js が読む（UL_BOARD_SIZE=3固定）。
# 実験変数:
#   EXP    実験名（dir/prefix に使用）
#   ROUNDS ラウンド数（既定30）
set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> JULIUS root

export UL_BOARD_SIZE=3
EXP="${EXP:?set EXP}"
ROUNDS="${ROUNDS:-30}"
DIR="research/alphazero/data/3x3_${EXP}"
PREFIX="az3_${EXP}_r"
SP_GAMES=500
SP_SIMS=160
PROCS=10

echo "=== 3x3 experiment ${EXP} | white=${UL_MAX_WHITE:-def} black=${UL_MAX_BLACK:-def} unlock=${UL_ELIM_UNLOCK:-both} | rounds=${ROUNDS} ==="
rm -rf "$DIR"; rm -f research/models/${PREFIX}*.json research/models/${PREFIX}*.bin research/models/${PREFIX}*.golden.json 2>/dev/null || true

for ((r=1; r<=ROUNDS; r++)); do
  if [[ $r -eq 1 ]]; then
    SEED_ARGS="--seed-net random --fresh-round1 true"
  else
    SEED_ARGS=""
  fi
  node research/alphazero/selfplay/runRound.mjs \
    --dir "$DIR" $SEED_ARGS \
    --size 3 --sp-games $SP_GAMES --sp-sims $SP_SIMS --procs $PROCS \
    --sample 200000 --epochs 10 --capacity 1500000 \
    --hidden1 96 --hidden2 24 --hidden3 24 --stack-encoder-dim 16 \
    --net-prefix "$PREFIX" --bench-every 0 \
    2>&1 | grep -E "selfplay metrics|ADVANCE" | sed "s/^/[${EXP}] /"
done

# ── 最終評価: 学習済み最終ネットで大量自己対戦し、先手(白)/後手(黒)勝率を安定推定 ──
FINAL_NET=$(node -e "const s=require('./${DIR}/loop_manifest.json');process.stdout.write(s.net)")
echo "[${EXP}] final net = ${FINAL_NET}  → running final self-play eval (2000 games, sims 200)"
node research/alphazero/selfplay/selfPlayWorker.mjs \
  --net "$FINAL_NET" --size 3 --games 2000 --sims 200 --seed 999 --augment 1 \
  --out "${DIR}/final_eval.jsonl" 2>&1 | grep -E "result="
echo "[${EXP}] DONE. per-round trajectory in ${DIR}/rounds.log ; final eval meta in ${DIR}/final_eval.meta.json"
