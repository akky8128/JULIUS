#!/usr/bin/env bash
# runMany.sh — 継続更新型 自己改善ループを START から END まで連続実行する（1プロセスで複数ラウンド）。
# 毎ラウンドの呼び出し往復オーバーヘッドを避けるための駆動スクリプト。gating は無く毎周 net を無条件advance。
# 使い方: ./runMany.sh <start_round> <end_round_inclusive>
set -euo pipefail
cd "$(dirname "$0")/../../.."  # -> JULIUS root

START="$1"
END="$2"
# 環境変数で上書き可能（既定は継続runの loop2 設定）。EPOCHS はモード崩壊防止のため
# 標準AZ流に少数へ。過学習(25ep)は方策の尖鋭化→自己対戦の決定化→崩壊を招くため。
DIR="${DIR:-tools/alphazero/data/loop3}"
SEED_NET="${SEED_NET:-models/genAZc_r10.json}"  # round1 のみ使用（以降はmanifestのnet）
PREFIX="${PREFIX:-genAZd_r}"
EPOCHS="${EPOCHS:-4}"
CAPACITY="${CAPACITY:-1500000}"
SIMS="${SIMS:-100}"        # 自己対戦MCTS sims。低すぎると探索が事前分布を修正できず崩壊する
SPGAMES="${SPGAMES:-600}"  # 1周の自己対戦局数
# loop8 スケール施策のための環境変数（既定は現行値＝挙動不変）
HIDDEN1="${HIDDEN1:-96}"
HIDDEN2="${HIDDEN2:-24}"
HIDDEN3="${HIDDEN3:-24}"
STACK_ENC="${STACK_ENC:-16}"
FRESH_ROUND1="${FRESH_ROUND1:-false}"  # round1で warm-start を切る（拡大ネット時 true）
BENCH_GAMES="${BENCH_GAMES:-40}"
BENCH_EVERY="${BENCH_EVERY:-5}"
PROCS="${PROCS:-10}"
SAMPLE="${SAMPLE:-400000}"

mkdir -p "$DIR"  # ログ追記(>>)前にディレクトリを用意（set -e で即死するのを防ぐ）

for ((r=START; r<=END; r++)); do
  echo "=== round $r starting $(date '+%H:%M:%S') ===" >> "$DIR/runMany.log"
  node research/alphazero/selfplay/runRound.mjs --round "$r" --dir "$DIR" \
    --seed-net "$SEED_NET" --net-prefix "$PREFIX" \
    --sp-games "$SPGAMES" --sp-sims "$SIMS" --procs "$PROCS" \
    --sample "$SAMPLE" --epochs "$EPOCHS" \
    --hidden1 "$HIDDEN1" --hidden2 "$HIDDEN2" --hidden3 "$HIDDEN3" \
    --stack-encoder-dim "$STACK_ENC" --fresh-round1 "$FRESH_ROUND1" \
    --capacity "$CAPACITY" --bench-every "$BENCH_EVERY" --bench-games "$BENCH_GAMES" >> "$DIR/runMany.log" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "=== round $r FAILED (exit $status) at $(date '+%H:%M:%S') — stopping ===" >> "$DIR/runMany.log"
    exit $status
  fi
  echo "=== round $r done $(date '+%H:%M:%S') ===" >> "$DIR/runMany.log"
done
echo "=== all rounds $START..$END complete $(date '+%H:%M:%S') ===" >> "$DIR/runMany.log"
