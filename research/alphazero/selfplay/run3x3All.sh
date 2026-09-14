#!/usr/bin/env bash
# run3x3All.sh — 3実験を順番に(scratchから)実行。CPU/MPS競合を避けるため逐次。
set -uo pipefail
cd "$(dirname "$0")/../../.."
ROUNDS="${ROUNDS:-30}"
export ROUNDS

echo "########## $(date) START 3x3 first-player-advantage experiments ##########"

echo "########## EXP1: white=4 black=4 unlock=both ##########"
EXP=exp1 UL_MAX_WHITE=4 UL_MAX_BLACK=4 UL_ELIM_UNLOCK=both bash research/alphazero/selfplay/run3x3Exp.sh

echo "########## EXP2: white=4 black=5 unlock=both ##########"
EXP=exp2 UL_MAX_WHITE=4 UL_MAX_BLACK=5 UL_ELIM_UNLOCK=both bash research/alphazero/selfplay/run3x3Exp.sh

echo "########## EXP3: white=4 black=5 unlock=self ##########"
EXP=exp3 UL_MAX_WHITE=4 UL_MAX_BLACK=5 UL_ELIM_UNLOCK=self bash research/alphazero/selfplay/run3x3Exp.sh

echo "########## $(date) ALL DONE ##########"
