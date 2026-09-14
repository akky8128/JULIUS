#!/usr/bin/env bash
# run4x4Exp.sh — 4x4本番盤 先手有利/駒数ハンデ検証の1実験を scratch から回す。
#
# 環境変数（ルール, gameLogic.js が読む）:
#   UL_MAX_WHITE / UL_MAX_BLACK / UL_ELIM_UNLOCK(both|self)  ※UL_BOARD_SIZE=4
# 実験変数:
#   EXP    実験名（dir/prefix に使用）
#   ROUNDS ラウンド数（既定40）
set -uo pipefail
cd "$(dirname "$0")/../../.."

export UL_BOARD_SIZE=4
EXP="${EXP:?set EXP}"
ROUNDS="${ROUNDS:-25}"
# 強いgen83(8/8対称)からwarm-start。4x4はscratchだと100+ラウンド必要なため、
# 強い事前分布から新ルールへ適応させ、勝率トレンドを記録する。
SEED_NET="${SEED_NET:-models/genAZ8_r83.json}"
DIR="research/alphazero/data/4x4_${EXP}"
PREFIX="az4_${EXP}_r"
SP_GAMES="${SP_GAMES:-300}"
SP_SIMS="${SP_SIMS:-140}"
PROCS=10

echo "=== 4x4 experiment ${EXP} | white=${UL_MAX_WHITE:-def} black=${UL_MAX_BLACK:-def} unlock=${UL_ELIM_UNLOCK:-both} | rounds=${ROUNDS} spGames=${SP_GAMES} sims=${SP_SIMS} warmStart=${SEED_NET} ==="
rm -rf "$DIR"; rm -f research/models/${PREFIX}*.json research/models/${PREFIX}*.bin research/models/${PREFIX}*.golden.json 2>/dev/null || true

for ((r=1; r<=ROUNDS; r++)); do
  if [[ $r -eq 1 ]]; then SEED_ARGS="--seed-net ${SEED_NET}"; else SEED_ARGS=""; fi
  node research/alphazero/selfplay/runRound.mjs \
    --dir "$DIR" $SEED_ARGS \
    --size 4 --sp-games $SP_GAMES --sp-sims $SP_SIMS --procs $PROCS \
    --sample 300000 --epochs 12 --capacity 2500000 \
    --hidden1 160 --hidden2 48 --hidden3 48 --stack-encoder-dim 32 \
    --net-prefix "$PREFIX" --bench-every 0 \
    2>&1 | grep -E "selfplay metrics|ADVANCE" | sed "s/^/[${EXP}] /"
done

FINAL_NET=$(node -e "const s=require('./${DIR}/loop_manifest.json');process.stdout.write(s.net)")
echo "[${EXP}] final net = ${FINAL_NET}  → final self-play eval (1200 games, sims 160, 4 parallel shards)"
# 並列4シャードで評価を高速化（単一プロセスだと4x4は非常に遅い）。
for i in 0 1 2 3; do
  node research/alphazero/selfplay/selfPlayWorker.mjs \
    --net "$FINAL_NET" --size 4 --games 300 --sims 160 --seed $((900+i)) --augment 1 \
    --out "${DIR}/final_eval_s${i}.jsonl" 2>&1 | grep -E "result=" &
done
wait
node -e "
let w=0,b=0,d=0,g=0,p=0;
for(let i=0;i<4;i++){const m=require('./${DIR}/final_eval_s'+i+'.meta.json');const r=m.resultDistribution;w+=r.white;b+=r.black;d+=r.draw;g+=m.numGames;p+=m.avgPlies*m.numGames;}
const fs=require('fs');fs.writeFileSync('./${DIR}/final_eval.meta.json',JSON.stringify({resultDistribution:{white:w,black:b,draw:d},numGames:g,avgPlies:p/g},null,2));
console.log('[${EXP}] FINAL  games='+g+' WHITE='+(100*w/g).toFixed(1)+'% BLACK='+(100*b/g).toFixed(1)+'% DRAW='+(100*d/g).toFixed(1)+'%');
"
echo "[${EXP}] DONE. trajectory: ${DIR}/rounds.log ; final eval: ${DIR}/final_eval.meta.json"
