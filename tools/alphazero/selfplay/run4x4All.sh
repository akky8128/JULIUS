#!/usr/bin/env bash
# run4x4All.sh — 4x4 学習3実験(4B/4C/4D)を順番に実行。gen83からwarm-start。
# 4A(gen83そのまま評価)は学習不要のため別途 selfPlayWorker で実施済み。
set -uo pipefail
cd "$(dirname "$0")/../../.."
ROUNDS="${ROUNDS:-25}"; export ROUNDS

echo "########## $(date) START 4x4 experiments (warm-start from gen83) ##########"

echo "########## 4B: white=7 black=8 unlock=both ##########"
EXP=4B UL_MAX_WHITE=7 UL_MAX_BLACK=8 UL_ELIM_UNLOCK=both bash tools/alphazero/selfplay/run4x4Exp.sh

echo "########## 4C: white=8 black=8 unlock=self ##########"
EXP=4C UL_MAX_WHITE=8 UL_MAX_BLACK=8 UL_ELIM_UNLOCK=self bash tools/alphazero/selfplay/run4x4Exp.sh

echo "########## 4D: white=7 black=8 unlock=self ##########"
EXP=4D UL_MAX_WHITE=7 UL_MAX_BLACK=8 UL_ELIM_UNLOCK=self bash tools/alphazero/selfplay/run4x4Exp.sh

echo "########## $(date) 4x4 ALL DONE ##########"
