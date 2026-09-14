/**
 * migrate_scalars.mjs — 既存データ(4スカラー)を新特徴(9スカラー)へ移行する。
 *
 * 各レコードの indices は genData のスキャン順(r→c→スタック下から上)で
 * 1コマ1indexが push されており、配列順がスタックの底→頂の順序と色を厳密に保持する
 * (depth clip でスロットが飽和しても配列順で真の順序が復元できる)。
 * よって盤面を厳密に再構成でき、extractFeatures を再実行して9スカラーを得る。
 *
 * 検証: 再構成盤から得た indices が元の indices と完全一致し、かつ scalars[0..3] が
 * 元と一致することを全レコードで確認する(不一致は致命的として停止)。
 *
 * 使い方: node research/nnue/migrate_scalars.mjs <in.jsonl> <out.jsonl>
 */
import { createInterface } from "node:readline";
import { createReadStream, createWriteStream } from "node:fs";
import { extractFeatures, MAX_DEPTH } from "../../core/nnue/features.js";
import { maxSummonsFor } from "../../core/gameLogic.js";

const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) { console.error("usage: migrate_scalars.mjs <in> <out>"); process.exit(1); }

const SLOTS = MAX_DEPTH * 2; // 1セルあたりの one-hot スロット数

function reconstruct(rec) {
  const size = rec.size;
  const pov = rec.pov;
  const opp = pov === "white" ? "black" : "white";
  const board = Array.from({ length: size }, () => Array.from({ length: size }, () => []));
  // indices は配列順で (cell, 底→頂) を保持。color: 0=pov, 1=opp。
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
  return { board, summonCounts, currentPlayer: pov, boardSize: size };
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

const rl = createInterface({ input: createReadStream(inPath), crlfDelay: Infinity });
const w = createWriteStream(outPath);
let n = 0, mismatchIdx = 0, mismatchScalar = 0;
const EPS = 1e-6;

for await (const line of rl) {
  if (!line.trim()) continue;
  const rec = JSON.parse(line);
  const state = reconstruct(rec);
  const f = extractFeatures(state, rec.pov);
  // 検証: indices 完全一致
  if (!arraysEqual(f.indices, rec.indices)) mismatchIdx++;
  // 検証: 既存4スカラー一致
  for (let k = 0; k < 4; k++) {
    if (Math.abs(f.scalars[k] - rec.scalars[k]) > EPS) { mismatchScalar++; break; }
  }
  const out = { ...rec, scalars: f.scalars };
  w.write(JSON.stringify(out) + "\n");
  n++;
  if (n % 500000 === 0) console.error(`  ...${n} 件 (idx不一致=${mismatchIdx} scalar不一致=${mismatchScalar})`);
}
await new Promise((res) => w.end(res));
console.error(`完了: ${n} 件, indices不一致=${mismatchIdx}, scalar不一致=${mismatchScalar}`);
if (mismatchIdx > 0 || mismatchScalar > 0) { console.error("警告: 不一致あり — 再構成が非厳密"); process.exit(2); }
console.error("検証OK: 全件で再構成が厳密一致");
