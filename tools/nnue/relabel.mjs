/**
 * relabel.mjs — 既存教師データの score を「より深い探索」で振り直す（フル再ラベル）。
 *
 * 目的（Stage 0 実験）: gen137 が頭打ちなのは「浅い探索(depth3/150ms)の教師ラベルを
 * 学習し切ったから」という仮説の統制検証。同一局面集合を、より強い評価器(gen137)＋
 * より深い探索でラベルし直し、教師信号の質だけを変えて再学習する。
 *
 * 各レコードの indices はスキャン順(cell, スタック底→頂, 色)を厳密に保持するため
 * 盤面を厳密再構成できる(migrate_scalars.mjs と同じ復元ロジック)。復元した state に
 * findBestTurn を1回かけてスコアを得る。score/scalars/result は D4 対称(sym)で不変
 * なので、**同一 pid(=sym 0..7 の8枚)ではスコアを1回だけ計算して使い回す**(8倍高速)。
 * sym 兄弟はファイル上で連続するため、直前 pid のスコアをキャッシュするだけでよい
 * (シャード境界で最大 N-1 回だけ再計算が起きるが無視できる)。
 *
 * result(対局結果)と indices/scalars/sym/pov/ply は一切変更しない。score のみ差し替える。
 *
 * 使い方:
 *   node tools/nnue/relabel.mjs --in <in.jsonl> --out <out.jsonl> \
 *     --depth 5 --timeMs 1500 [--net models/gen137.json] [--limit N]
 *   --net 省略時は heuristic(WEIGHTS) 評価器で探索する。
 *   --limit はベンチ用に先頭 N レコードだけ処理する。
 */
import fs from "node:fs";
import { createInterface } from "node:readline";
import { createReadStream, createWriteStream } from "node:fs";
import { findBestTurn } from "../../js/ai/search.js";
import { WEIGHTS } from "../../js/ai/evaluate.js";
import { loadNetwork } from "../../js/ai/nnue/network.js";
import { extractFeatures, MAX_DEPTH } from "../../js/ai/nnue/features.js";
import { maxSummonsFor, normalizeBoard } from "../../js/gameLogic.js";

const SLOTS = MAX_DEPTH * 2; // 1セルあたりの one-hot スロット数（深さ8 × 色2）

/** genData.mjs と同一の決定論 PRNG（探索のタイブレーク用）。 */
function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6D2B79F5) >>> 0;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function parseArgs() {
  const a = { depth: 5, timeMs: 1500, net: null, limit: Infinity, seed: 12345 };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    if (k === "--in") a.in = argv[++i];
    else if (k === "--out") a.out = argv[++i];
    else if (k === "--depth") a.depth = parseInt(argv[++i], 10);
    else if (k === "--timeMs") a.timeMs = parseInt(argv[++i], 10);
    else if (k === "--net") a.net = argv[++i];
    else if (k === "--limit") a.limit = parseInt(argv[++i], 10);
    else if (k === "--seed") a.seed = parseInt(argv[++i], 10);
  }
  if (!a.in || !a.out) {
    console.error("usage: relabel.mjs --in <in> --out <out> --depth D --timeMs T [--net PATH] [--limit N]");
    process.exit(1);
  }
  return a;
}

/** レコードの indices/scalars から state を厳密再構成する（migrate_scalars と同ロジック）。 */
function reconstruct(rec) {
  const size = rec.size;
  const pov = rec.pov;
  const opp = pov === "white" ? "black" : "white";
  const board = Array.from({ length: size }, () => Array.from({ length: size }, () => []));
  for (const idx of rec.indices) {
    const cell = Math.floor(idx / SLOTS);
    const rem = idx - cell * SLOTS;
    const color = rem % 2;
    const r = Math.floor(cell / size);
    const c = cell % size;
    board[r][c].push(color === 0 ? pov : opp);
  }
  const maxS = maxSummonsFor(size);
  const summonCounts = {};
  summonCounts[pov] = Math.round(maxS - rec.scalars[0] * maxS);
  summonCounts[opp] = Math.round(maxS - rec.scalars[1] * maxS);
  return { board: normalizeBoard(board, size), summonCounts, currentPlayer: pov, boardSize: size };
}

async function main() {
  const args = parseArgs();
  let evalFn = null;
  if (args.net) {
    const net = await loadNetwork(args.net);
    evalFn = (s, p) => net.evaluateState(s, p);
  }
  const searchOptions = (rng) => evalFn
    ? { maxDepth: args.depth, timeBudgetMs: args.timeMs, evalFn, rng }
    : { maxDepth: args.depth, timeBudgetMs: args.timeMs, weights: WEIGHTS, rng };

  const rng = mulberry32(args.seed >>> 0);
  const rl = createInterface({ input: createReadStream(args.in), crlfDelay: Infinity });
  const w = createWriteStream(args.out);

  let n = 0, searched = 0, reused = 0;
  let lastPid = null, lastScore = null;
  const t0 = Date.now();

  for await (const line of rl) {
    if (!line.trim()) continue;
    if (n >= args.limit) break;
    const rec = JSON.parse(line);
    let newScore;
    if (rec.pid === lastPid) {
      newScore = lastScore;
      reused++;
    } else {
      const state = reconstruct(rec);
      newScore = findBestTurn(state, searchOptions(rng)).score;
      lastPid = rec.pid;
      lastScore = newScore;
      searched++;
    }
    rec.score = newScore;
    w.write(JSON.stringify(rec) + "\n");
    n++;
    if (searched > 0 && searched % 5000 === 0 && reused % 8 === 7) {
      const dt = (Date.now() - t0) / 1000;
      process.stderr.write(`[relabel] recs=${n} searched=${searched} (${(searched / dt).toFixed(1)} pos/s) reused=${reused}\n`);
    }
  }
  w.end();
  const dt = (Date.now() - t0) / 1000;
  console.log(`[relabel] DONE in=${args.in} out=${args.out} depth=${args.depth} timeMs=${args.timeMs} net=${args.net || "heuristic"}`);
  console.log(`[relabel] records=${n} searched=${searched} reused=${reused} elapsed=${dt.toFixed(1)}s rate=${(searched / dt).toFixed(1)} pos/s`);
}

main();
