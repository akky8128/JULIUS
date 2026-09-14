#!/usr/bin/env bash
# run3x3ElimStart.sh — 3x3盤・排除をゲーム開始から解禁(召喚/排除ノード分離)で
# scratchからAlphaZeroを学習し先手後手勝率を記録する2実験を順次実行。
#   E1: 白4 黒4    E2: 白4 黒5
# ルール: UL_ELIM_UNLOCK=always（開始から排除可） + UL_SEP_ELIM=1（ポリシーノード分離, NUM=97）
set -uo pipefail
cd "$(dirname "$0")/../../.."
ROUNDS="${ROUNDS:-30}"

run_one () {
  local EXP="$1" MW="$2" MB="$3"
  local DIR="research/alphazero/data/3x3elim_${EXP}"
  local PREFIX="az3e_${EXP}_r"
  echo "=== 3x3 elim-from-start ${EXP} | white=${MW} black=${MB} | sep-nodes | rounds=${ROUNDS} ==="
  rm -rf "$DIR"; rm -f research/models/${PREFIX}*.json research/models/${PREFIX}*.bin research/models/${PREFIX}*.golden.json 2>/dev/null || true
  for ((r=1; r<=ROUNDS; r++)); do
    if [[ $r -eq 1 ]]; then SEED_ARGS="--seed-net random --fresh-round1 true"; else SEED_ARGS=""; fi
    UL_BOARD_SIZE=3 UL_SEP_ELIM=1 UL_ELIM_UNLOCK=always UL_MAX_WHITE=$MW UL_MAX_BLACK=$MB \
    node research/alphazero/selfplay/runRound.mjs \
      --dir "$DIR" $SEED_ARGS \
      --size 3 --sp-games 500 --sp-sims 160 --procs 10 \
      --sample 200000 --epochs 10 --capacity 1500000 \
      --hidden1 96 --hidden2 24 --hidden3 24 --stack-encoder-dim 16 \
      --net-prefix "$PREFIX" --bench-every 0 \
      2>&1 | grep -E "selfplay metrics|ADVANCE" | sed "s/^/[${EXP}] /"
  done
  local FINAL_NET=$(node -e "const s=require('./${DIR}/loop_manifest.json');process.stdout.write(s.net)")
  echo "[${EXP}] final net=${FINAL_NET} → final eval (2000 games, sims 200)"
  UL_BOARD_SIZE=3 UL_SEP_ELIM=1 UL_ELIM_UNLOCK=always UL_MAX_WHITE=$MW UL_MAX_BLACK=$MB \
  node research/alphazero/selfplay/selfPlayWorker.mjs \
    --net "$FINAL_NET" --size 3 --games 2000 --sims 200 --seed 999 --augment 1 \
    --out "${DIR}/final_eval.jsonl" 2>&1 | grep -E "result="
  echo "[${EXP}] DONE."
}

echo "########## $(date) START 3x3 eliminate-from-start experiments ##########"
run_one E1 4 4
run_one E2 4 5
echo "########## $(date) ALL DONE ##########"
