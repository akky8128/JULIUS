#!/usr/bin/env node
/**
 * runRound.mjs — 真のAlphaZero型 自己改善ループの「1ラウンド」駆動（実計算・並列）。
 *
 * ★継続更新型（gating無し・単一ネット継続更新）★
 * 1ラウンド = 最新ネットで並列自己対戦 → リプレイバッファ追記 → バッチsample →
 *   train.py(warm-start継続学習, --init-bin=最新net.bin, 2ヘッド) → 新ネット genAZ_r{N} を
 *   **無条件に最新ネットとして採用**（採否判定・champion概念・2標本z・昇格ゲートは一切無し）。
 *
 * champion/best/acceptStreak といった gating 状態は廃止。manifest は最新ネット net・周回 round・
 * 各周のtrain統計 history のみを持つ。毎周 self-play→buffer→train→advance だけを行う。
 *
 * このスクリプトは1ラウンドだけ実行して return する（無人無限ループではない）。呼び出し側
 * （オーケストレータ）が繰り返し再実行する。並列度は --procs。
 *
 * round1 のときだけ --seed-net（既定 research/models/genAZ001.json）を最新ネットとして使い、
 * 以降は manifest.net を使う。--bench-every K 指定時は K周ごとに現行netで benchmark.mjs を実行。
 *
 * 使い方:
 *   node research/alphazero/selfplay/runRound.mjs \
 *     --dir research/alphazero/data/loop2 --seed-net research/models/genAZ001.json \
 *     --sp-games 1000 --sp-sims 100 --procs 10 --sample 400000 --epochs 25 --bench-every 5
 */

import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

import { openBuffer, appendShard, saveManifest, sampleToFile, sampleStratifiedToFile, bufferStats } from "./replayBuffer.mjs";
import { DRAW_Z } from "./selfPlayWorker.mjs";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..", "..");

function arg(k, d) { const i = process.argv.indexOf(k); return i >= 0 ? process.argv[i + 1] : d; }

/** 子プロセスを spawn し、終了コードと収集した stdout を Promise で返す。 */
function run(cmd, args, { capture = false, label = "" } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd: ROOT, stdio: capture ? ["ignore", "pipe", "inherit"] : "inherit" });
    let out = "";
    if (capture) child.stdout.on("data", (d) => { out += d.toString(); });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) reject(new Error(`${label || cmd} exited ${code}`));
      else resolve(out);
    });
  });
}

/** N 個のタスク（() => Promise）を最大 concurrency で並列実行。 */
async function pMapLimit(tasks, concurrency) {
  const results = new Array(tasks.length);
  let idx = 0;
  async function worker() {
    while (idx < tasks.length) {
      const cur = idx++;
      results[cur] = await tasks[cur]();
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, tasks.length) }, worker));
  return results;
}

function splitGames(total, procs) {
  const base = Math.floor(total / procs);
  const rem = total % procs;
  return Array.from({ length: procs }, (_, i) => base + (i < rem ? 1 : 0)).filter((g) => g > 0);
}

const manifestPath = (dir) => path.join(dir, "loop_manifest.json");

/**
 * 継続更新ループの状態を読み込む（resume）。
 * shape: { net: <現行単一ネットのパス>, round: <int>, history: [ ... ] }
 * 無ければ { net: seedNet, round: 0, history: [] }（round1 で seedNet を使う）。
 */
function loadState(dir, seedNet) {
  const mp = manifestPath(dir);
  if (fs.existsSync(mp)) {
    const s = JSON.parse(fs.readFileSync(mp, "utf8"));
    // 継続更新schemaへ最低限の整合（古い gated manifest を読んでも net を復元）。
    if (!s.net) s.net = s.champion || seedNet;
    if (!Array.isArray(s.history)) s.history = [];
    if (typeof s.round !== "number") s.round = 0;
    return { net: s.net, round: s.round, history: s.history };
  }
  return { net: seedNet, round: 0, history: [] };
}

function saveState(dir, state) {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(manifestPath(dir), JSON.stringify(state, null, 2), "utf8");
}

async function main() {
  const t0 = Date.now();
  const dir = arg("--dir", "research/alphazero/data/loop2");
  const seedNet = arg("--seed-net", "research/models/genAZ001.json");
  const spGames = parseInt(arg("--sp-games", "1000"), 10);
  const spSims = parseInt(arg("--sp-sims", "100"), 10);
  const procs = parseInt(arg("--procs", "10"), 10);
  const sample = parseInt(arg("--sample", "400000"), 10);
  const epochs = parseInt(arg("--epochs", "25"), 10);
  const capacityStates = parseInt(arg("--capacity", "2000000"), 10);
  const benchEvery = parseInt(arg("--bench-every", "0"), 10);
  const benchGames = parseInt(arg("--bench-games", "40"), 10);
  // モデル出力タグの接頭辞。既定 genAZ_r。継続runでは別接頭辞(例 genAZc_r)を指定して
  // 昨夜のgated run産物(research/models/genAZ_r3..r65)を上書きしないようにする。
  const netPrefix = arg("--net-prefix", "genAZ_r");
  // loop6 A1: 温度スケジュール（ply基準・序盤多様化＋中盤の貪欲固執抑制）
  const openingPlies = parseInt(arg("--opening-plies", "6"), 10);
  const sustainedTemp = parseFloat(arg("--sustained-temp", "0.25"));
  // loop6 C1: 決着局オーバーサンプリング。0以下で無効（従来の一様サンプリング）。
  const decisiveFrac = parseFloat(arg("--decisive-frac", "0.5"));
  // loop8 ③ネット拡大: train.py へ渡す隠れ層/stack-encoder 次元（既定は現行 96/24/24, 16）。
  const hidden1 = arg("--hidden1", "96");
  const hidden2 = arg("--hidden2", "24");
  const hidden3 = arg("--hidden3", "24");
  const stackEncoderDim = arg("--stack-encoder-dim", "16");
  // loop8: round1 で warm-start を切る（seed-net と学習アーキの形状が異なる場合に必須）。
  const freshRound1 = arg("--fresh-round1", "false") === "true";
  // 3x3先手有利検証: 盤面サイズを self-play に渡す（環境変数 UL_BOARD_SIZE がフォールバック）。
  const size = arg("--size", process.env.UL_BOARD_SIZE || "4");

  const state = loadState(dir, seedNet);
  const round = (arg("--round", null) !== null) ? parseInt(arg("--round", "1"), 10) : state.round + 1;
  const currentNet = state.net; // round1 => seedNet, 以降 => 前周の {prefix}{round-1}
  const netTag = `${netPrefix}${round}`;
  const newNetJson = path.join("research", "models", `${netTag}.json`);

  const dataDir = path.join(dir, "shards");
  const bufferDir = path.join(dir, "buffer");
  const batchPath = path.join(dir, `batch_r${round}.jsonl`);
  fs.mkdirSync(dataDir, { recursive: true });

  const log = (m) => console.log(`[r${round}] ${m}`);
  log(`net(latest)=${currentNet} spGames=${spGames} spSims=${spSims} procs=${procs} sample=${sample} epochs=${epochs}`);

  // ── 1. 並列自己対戦（最新ネット）───────────────────────────────
  const tSelf = Date.now();
  const shardGames = splitGames(spGames, procs);
  const shardPaths = shardGames.map((_, i) => path.join(dataDir, `sp_r${round}_s${i}.jsonl`));
  await pMapLimit(shardGames.map((g, i) => () => run("node", [
    "research/alphazero/selfplay/selfPlayWorker.mjs",
    "--net", currentNet, "--games", String(g), "--sims", String(spSims),
    "--size", String(size),
    "--seed", String(1000 * round + i), "--augment", "8",
    "--pidOffset", String(round * 10_000_000 + i * 1_000_000),
    "--opening-plies", String(openingPlies), "--sustained-temp", String(sustainedTemp),
    "--out", shardPaths[i],
  ], { label: `selfplay s${i}` })), procs);
  const selfSec = ((Date.now() - tSelf) / 1000).toFixed(1);
  log(`self-play done ${selfSec}s (${shardGames.length} shards, ${spGames} games)`);

  // ── loop6 D: 崩壊監視指標（今ラウンドの自己対戦シャード meta を集計）──
  let mgWhite = 0, mgBlack = 0, mgDraw = 0, mgPlies = 0, mgSummon = 0, mgGames = 0;
  for (let i = 0; i < shardPaths.length; i++) {
    const mp = shardPaths[i].replace(/\.jsonl$/, ".meta.json");
    try {
      const m = JSON.parse(fs.readFileSync(mp, "utf8"));
      const rd = m.resultDistribution || {};
      const g = m.numGames || 0;
      mgWhite += rd.white || 0; mgBlack += rd.black || 0; mgDraw += rd.draw || 0;
      mgPlies += (m.avgPlies || 0) * g; mgSummon += (m.avgSummon || 0) * g; mgGames += g;
    } catch { /* meta 欠損は無視 */ }
  }
  const drawRate = mgGames > 0 ? mgDraw / mgGames : 0;
  const avgPlies = mgGames > 0 ? mgPlies / mgGames : 0;
  const avgSummon = mgGames > 0 ? mgSummon / mgGames : 0;
  const spMetrics = { games: mgGames, white: mgWhite, black: mgBlack, draw: mgDraw, drawRate, avgPlies, avgSummon };
  log(`selfplay metrics: drawRate=${(drawRate * 100).toFixed(1)}% decisive=${mgWhite + mgBlack}/${mgGames} avgPlies=${avgPlies.toFixed(1)} avgSummon=${avgSummon.toFixed(2)}`);

  // ── 2. リプレイバッファ追記 ──────────────────────────────────
  const buffer = openBuffer(bufferDir, { capacityStates });
  buffer.manifest.round = round;
  for (const sp of shardPaths) appendShard(buffer, sp, { round });
  saveManifest(buffer);
  const bstats = bufferStats(buffer);
  log(`buffer: totalStates=${bstats.totalStates} shards=${bstats.numShards}`);

  // ── 3. バッチ sample（loop6 C1: 決着局オーバーサンプリング）──────────
  let sampled, sampleInfo = null;
  if (decisiveFrac > 0) {
    const r = sampleStratifiedToFile(buffer, sample, batchPath, Math.random, { decisiveFrac, drawZ: DRAW_Z });
    sampled = r.sampled; sampleInfo = r;
    log(`sampled ${sampled} (decisive=${r.decisive} draw=${r.draw}; avail dec=${r.decisiveAvail} draw=${r.drawAvail}) → ${path.basename(batchPath)}`);
  } else {
    ({ sampled } = sampleToFile(buffer, sample, batchPath, Math.random));
    log(`sampled ${sampled} records (uniform) → ${path.basename(batchPath)}`);
  }

  // ── 4. train.py（warm-start継続学習, 2ヘッド）──────────────
  // 単一ネットの継続学習: 最新ネットの重みから warm-start し、蓄積リプレイバッファで学習を
  // 継続する。--init-bin は byte-identical round-trip で検証済み。学習アーキは現行と同一
  // （96/24/24 + stack-encoder-dim 16 + policy-head, turn-context-dim 26 は train.py 既定）。
  const currentBin = currentNet.replace(/\.json$/, ".bin");
  const tTrain = Date.now();
  // round1 で freshRound1 の場合のみ warm-start(--init-bin) を省いてスクラッチ学習する
  // （拡大ネット等で seed-net と学習アーキの形状が異なると .bin を読めないため）。
  const useInitBin = !(freshRound1 && round === 1);
  // 排除分離レイアウト時は policy 出力次元が 97 になる（UL_SEP_ELIM）。
  const numActions = (process.env.UL_SEP_ELIM === "1" || process.env.UL_SEP_ELIM === "true") ? "97" : "81";
  const trainArgs = [
    "research/nnue/train.py", "--data", batchPath, "--out-dir", "research/models", "--gen-tag", netTag,
    "--policy-head", "--num-actions", numActions,
    "--stack-encoder-dim", stackEncoderDim, "--hidden1", hidden1, "--hidden2", hidden2, "--hidden3", hidden3,
    "--epochs", String(epochs), "--patience", "6", "--batch-size", "512", "--lr", "1e-3",
    "--device", "auto", "--seed", String(round),
  ];
  if (useInitBin) trainArgs.push("--init-bin", currentBin);
  else log(`round1 fresh train (no warm-start): arch ${hidden1}/${hidden2}/${hidden3} stackEnc=${stackEncoderDim}`);
  await run("python3", trainArgs, { label: "train.py" });
  const trainSec = ((Date.now() - tTrain) / 1000).toFixed(1);
  log(`train done ${trainSec}s → ${newNetJson} (warm-start from ${path.basename(currentBin)})`);

  // 新ネットのtrain統計（valLoss / policy top-1）を成果物メタから読む（記録用）。
  let bestValLoss = null, top1 = null;
  try {
    const meta = JSON.parse(fs.readFileSync(path.join(ROOT, newNetJson), "utf8"));
    bestValLoss = meta?.train?.valLoss ?? null;
    top1 = meta?.train?.valPolicyTop1 ?? null;
  } catch { /* メタ読めなくても致命的ではない（記録がnullになるだけ） */ }

  // ── 5. 無条件advance（gating無し）──────────────────────────────
  // 採否判定・評価ゲート・2標本z・champion概念は一切無い。新ネットを最新ネットとして採用。
  const prevNet = state.net;
  state.net = newNetJson;
  state.round = round;
  state.history.push({
    round, net: newNetJson, prevNet,
    bestValLoss, top1,
    bufferStates: bstats.totalStates, bufferShards: bstats.numShards, sampled,
    spMetrics, sampleInfo,
    timings: { selfSec, trainSec },
  });
  saveState(dir, state);

  const totalSec = ((Date.now() - t0) / 1000).toFixed(1);
  fs.writeFileSync(path.join(dir, "last_round.json"), JSON.stringify({
    round, net: state.net, prevNet, bestValLoss, top1,
    bufferStates: bstats.totalStates, timings: { selfSec, trainSec, totalSec },
  }, null, 2));
  fs.appendFileSync(path.join(dir, "rounds.log"),
    `r${round} advance ${path.basename(prevNet)} → ${path.basename(state.net)} ` +
    `valLoss=${bestValLoss ?? "?"} top1=${top1 ?? "?"} bufStates=${bstats.totalStates} ` +
    `drawRate=${(drawRate * 100).toFixed(1)}% avgPlies=${avgPlies.toFixed(1)} avgSummon=${avgSummon.toFixed(2)} ` +
    `[self ${selfSec}s train ${trainSec}s total ${totalSec}s]\n`);

  log(`ADVANCE net → ${path.basename(state.net)} (unconditional) valLoss=${bestValLoss ?? "?"} top1=${top1 ?? "?"} (total ${totalSec}s)`);

  // ── 6. ベンチマーク（強さ追跡・記録のみ・採否には使わない）──────
  if (benchEvery > 0 && round % benchEvery === 0) {
    log(`bench: round%${benchEvery}==0 → benchmark.mjs (games=${benchGames})`);
    await run("node", [
      "research/alphazero/selfplay/benchmark.mjs",
      "--net", state.net, "--round", String(round), "--dir", dir,
      "--procs", String(procs), "--bench-games", String(benchGames),
    ], { label: "benchmark" }).catch((e) => log(`bench failed (non-fatal): ${e.message}`));
  }
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
