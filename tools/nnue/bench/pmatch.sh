#!/bin/zsh
# usage: pmatch.sh <white> <black> <totalGames> <shards> <outprefix> [baseSeed]
cd /Users/akky/Documents/HTML/JULIUS
W=$1; B=$2; TOT=$3; SH=$4; OUT=$5; BASE=${6:-1000}
per=$(( (TOT + SH - 1) / SH ))
pids=()
for i in $(seq 1 $SH); do
  seed=$(( BASE + i*97 ))
  node tools/selfplay.mjs --games $per --seed $seed --swap --white "$W" --black "$B" > ${OUT}.shard$i.txt 2>&1 &
  pids+=($!)
done
for p in $pids; do wait $p; done
node tools/nnue/bench/agg.mjs "$W" ${OUT}.shard*.txt | tee ${OUT}.agg.txt
