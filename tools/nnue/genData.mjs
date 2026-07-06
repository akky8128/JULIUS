#!/usr/bin/env node
/**
 * genData.mjs — NNUE 教師データ生成スクリプト（Step 2）
 *
 * 既存の探索エンジン（findBestTurn）同士を自己対戦させ、各局面の
 * 特徴（extractFeatures）・探索スコア（手番視点）・対局結果（手番視点）を
 * JSONL レコードとして書き出す。D4 対称変換によるデータ増強もサポートする。
 *
 * 設計方針:
 *   - buildDataset(config) は純粋関数（ファイルIOなし）。レコード配列と
 *     meta 情報を返すだけで、副作用を持たない。テスト容易性のため main() と
 *     分離している。
 *   - main() は CLI 引数のパース・ファイル書き込み・進捗表示を担当する。
 *
 * 使い方:
 *   node tools/nnue/genData.mjs [--games N] [--size N] [--depth N]
 *       [--timeMs N] [--maxPlies N] [--seed N] [--openingRandomPlies N]
 *       [--epsilon N] [--augment 1|8] [--out <path>]
 *
 * 大規模生成は重い処理になるため、疎通確認は小さな --games / --size に
 * 留めること。
 */

import fs from "node:fs";
import path from "node:path";

import { generateTurns } from "../../js/ai/moveGen.js";
import { findBestTurn } from "../../js/ai/search.js";
import { WEIGHTS } from "../../js/ai/evaluate.js";
import { loadNetwork } from "../../js/ai/nnue/network.js";
import { maxSummonsFor, normalizeBoard } from "../../js/gameLogic.js";
import {
  MAX_DEPTH,
  SCALAR_DIM,
  NUM_SYMMETRIES,
  featureDim,
  extractFeatures,
  transformState,
} from "../../js/ai/nnue/features.js";

// ───────────────────────── mulberry32 PRNG ─────────────────────────
// selfplay.mjs と同一実装（決定論的な派生のため踏襲する）。

/**
 * mulberry32 PRNG を生成する。
 * @param {number} seed
 * @returns {() => number} [0,1) の乱数を返す関数
 */
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ───────────────────────── 盤面ユーティリティ ─────────────────────────

/** 空の boardSize x boardSize 盤（各セルは空配列）を返す。 */
function emptyBoard(size) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => []));
}

/** プレイヤーを反転する。 */
function nextPlayer(p) {
  return p === "white" ? "black" : "white";
}

// ───────────────────────── 1局面分の一時レコード構築 ─────────────────────────

/**
 * 1つの元局面（sym=0）を D4 対称展開し、最終レコード配列へ push する。
 * scalars/score/result は sym 不変なので使い回し、indices のみ
 * transformState 後に extractFeatures で再計算する。
 *
 * 同一「元局面」から生成される全 sym レコードには同一の pid を付与する
 * （train.py 側でリークのないグループ分割を行うため）。
 *
 * @param {{ state: object, ply: number, score: number, mover: string }} tempRecord
 * @param {number} result - 0 | 0.5 | 1（mover 視点）
 * @param {number} augment - 1 または 8
 * @param {number} pid - この元局面の一意な整数ID
 * @param {Array<object>} outRecords - push 先の配列（呼び出し元で確保）
 */
function expandRecord(tempRecord, result, augment, pid, outRecords) {
  const { state, ply, score, mover } = tempRecord;
  const symCount = augment === 8 ? NUM_SYMMETRIES : 1;

  // sym=0 の特徴はすでに計算済み（呼び出し元が渡す）ケースもあるが、
  // ここでは一貫性のため sym=0 も transformState(state, 0) 経由で扱う。
  for (let sym = 0; sym < symCount; sym++) {
    const symState = sym === 0 ? state : transformState(state, sym);
    const { indices, scalars } = extractFeatures(symState, mover);

    outRecords.push({
      pid,
      size: state.boardSize,
      pov: mover,
      ply,
      sym,
      indices,
      scalars,
      score,
      result,
    });
  }
}

// ───────────────────────── 1ゲーム分の対局・記録ロジック ─────────────────────────

/**
 * 1ゲームを自己対戦し、一時レコード群（元局面, sym=0 情報のみ）を返す。
 * findBestTurn は探索エージェント（白黒とも）として毎 ply 呼び出す。
 * 着手選択は openingRandomPlies / epsilon に従いランダム手を混ぜるが、
 * スコアラベルは常に findBestTurn.score（手番視点の評価値）を使う。
 *
 * @param {{
 *   size: number, maxDepth: number, timeBudgetMs: number, maxPlies: number,
 *   openingRandomPlies: number, epsilon: number, evalFn?: (state, player) => number
 * }} config
 * @param {() => number} rng - このゲーム専用の PRNG
 * @param {() => number} searchRng - findBestTurn に渡す派生 PRNG
 * @returns {{ tempRecords: Array<object>, winner: string|null, plies: number }}
 */
function playAndRecordGame(config, rng, searchRng) {
  const { size, maxDepth, timeBudgetMs, maxPlies, openingRandomPlies, epsilon, evalFn } = config;

  let board = emptyBoard(size);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";
  let ply = 0;

  const tempRecords = [];
  let winner = null;

  while (ply < maxPlies) {
    const normBoard = normalizeBoard(board, size);
    const state = {
      board: normBoard,
      summonCounts: { ...summonCounts },
      currentPlayer,
      boardSize: size,
    };

    const candidates = generateTurns(state);
    if (candidates.length === 0) {
      // 現在手番に合法手がない → 相手の勝ち
      winner = nextPlayer(currentPlayer);
      break;
    }

    // 探索は必ず1回呼ぶ（スコアラベル取得のため）
    // evalFn が指定されていれば（NNUE 自己対戦ブートストラップ）それを使い、
    // 未指定なら従来通り heuristic（weights: WEIGHTS）を使う。
    const searchOptions = evalFn
      ? { maxDepth, timeBudgetMs, evalFn, rng: searchRng }
      : { maxDepth, timeBudgetMs, weights: WEIGHTS, rng: searchRng };
    const searchResult = findBestTurn(state, searchOptions);
    const score = searchResult.score;

    // 着手選択: openingRandomPlies 中 or epsilon 確率でランダム手
    const useRandom = ply < openingRandomPlies || rng() < epsilon;
    const chosen = useRandom
      ? candidates[Math.floor(rng() * candidates.length)]
      : (searchResult.turn || candidates[Math.floor(rng() * candidates.length)]);

    // 一時レコードとして局面（state）とラベル候補を保存
    tempRecords.push({
      state,
      ply,
      score,
      mover: currentPlayer,
    });

    // 着手適用
    board = normalizeBoard(chosen.board, size);
    summonCounts = chosen.summonCounts;
    currentPlayer = nextPlayer(currentPlayer);
    ply++;
  }

  return { tempRecords, winner, plies: ply };
}

// ───────────────────────── buildDataset（純粋関数） ─────────────────────────

/**
 * 教師データセットを生成する（純粋関数、ファイルIOなし）。
 * 同一 config（同一 seed 含む）であれば出力は完全に決定論的。
 *
 * @param {{
 *   games: number, size: number, maxDepth: number, timeBudgetMs: number,
 *   maxPlies: number, seed: number, openingRandomPlies: number,
 *   epsilon: number, augment: 1|8, evalFn?: (state, player) => number
 * }} config
 * @returns {{ records: Array<object>, meta: object }}
 */
export function buildDataset(config) {
  const {
    games,
    size,
    maxDepth,
    timeBudgetMs,
    maxPlies,
    seed,
    openingRandomPlies,
    epsilon,
    augment,
    evalFn,
  } = config;

  const masterRng = mulberry32(seed);
  const records = [];
  let nextPid = 0;

  let winCount = 0;
  let lossCountForMover = 0; // 使わないが将来のデバッグ用に残さず、resultDistributionで代用
  let winResults = 0;
  let drawResults = 0;
  let lossResults = 0;
  let scoreMin = Infinity;
  let scoreMax = -Infinity;
  let scoreSum = 0;
  let scoreCount = 0;
  const symCoverage = {};
  for (let s = 0; s < NUM_SYMMETRIES; s++) symCoverage[s] = 0;

  for (let g = 0; g < games; g++) {
    // ゲームごとに独立した PRNG を派生（selfplay.mjs と同じ方式）
    const gameSeed = (masterRng() * 0xffffffff) >>> 0;
    const gameRng = mulberry32(gameSeed);

    // 着手選択用 PRNG と探索用 PRNG を分離（再現性のため決定論的に派生）
    const moveRng = mulberry32((gameRng() * 0xffffffff) >>> 0);
    const searchRng = mulberry32((gameRng() * 0xffffffff) >>> 0);

    const { tempRecords, winner } = playAndRecordGame(
      { size, maxDepth, timeBudgetMs, maxPlies, openingRandomPlies, epsilon, evalFn },
      moveRng,
      searchRng
    );

    // 各一時レコードに result（mover 視点）を付与し、D4 展開する
    for (const tempRecord of tempRecords) {
      const { mover } = tempRecord;
      let result;
      if (winner === null) {
        result = 0.5;
        drawResults++;
      } else if (winner === mover) {
        result = 1;
        winResults++;
      } else {
        result = 0;
        lossResults++;
      }

      const before = records.length;
      const pid = nextPid++;
      expandRecord(tempRecord, result, augment, pid, records);
      for (let i = before; i < records.length; i++) {
        const rec = records[i];
        symCoverage[rec.sym] = (symCoverage[rec.sym] || 0) + 1;
        if (rec.score < scoreMin) scoreMin = rec.score;
        if (rec.score > scoreMax) scoreMax = rec.score;
        scoreSum += rec.score;
        scoreCount++;
      }
    }
  }

  const meta = {
    generatedAt: new Date().toISOString(),
    config,
    MAX_DEPTH,
    SCALAR_DIM,
    featureDim: featureDim(size),
    numGames: games,
    numRecords: records.length,
    numPositions: nextPid,
    generator: {
      evaluator: evalFn ? "nnue" : "heuristic",
      net: config.netPath || null,
    },
    resultDistribution: {
      win: winResults,
      draw: drawResults,
      loss: lossResults,
    },
    scoreStats: {
      min: scoreCount > 0 ? scoreMin : null,
      max: scoreCount > 0 ? scoreMax : null,
      mean: scoreCount > 0 ? scoreSum / scoreCount : null,
    },
    symCoverage,
  };

  return { records, meta };
}

// ───────────────────────── CLI 引数パース ─────────────────────────

/**
 * `--key value` 形式の CLI 引数をパースする。
 * @param {string[]} argv
 * @returns {object}
 */
function parseArgs(argv) {
  const args = {
    games: 200,
    size: 4,
    depth: 3,
    timeMs: 200,
    maxPlies: 200,
    seed: 1,
    openingRandomPlies: 6,
    epsilon: 0.1,
    augment: 8,
    out: "tools/nnue/data/gen000.jsonl",
    net: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const val = argv[i + 1];
    if (key === "--games") args.games = parseInt(val, 10);
    else if (key === "--size") args.size = parseInt(val, 10);
    else if (key === "--depth") args.depth = parseInt(val, 10);
    else if (key === "--timeMs") args.timeMs = parseInt(val, 10);
    else if (key === "--maxPlies") args.maxPlies = parseInt(val, 10);
    else if (key === "--seed") args.seed = parseInt(val, 10);
    else if (key === "--openingRandomPlies") args.openingRandomPlies = parseInt(val, 10);
    else if (key === "--epsilon") args.epsilon = parseFloat(val);
    else if (key === "--augment") args.augment = parseInt(val, 10);
    else if (key === "--out") args.out = val;
    else if (key === "--net") args.net = val;
  }
  return args;
}

/** `.jsonl` 拡張子を `.meta.json` に置き換える。 */
function metaPathFor(outPath) {
  const ext = path.extname(outPath);
  const base = ext ? outPath.slice(0, -ext.length) : outPath;
  return `${base}.meta.json`;
}

// ───────────────────────── main（CLI エントリポイント） ─────────────────────────

async function main() {
  const cliArgs = parseArgs(process.argv.slice(2));
  const config = {
    games: cliArgs.games,
    size: cliArgs.size,
    maxDepth: cliArgs.depth,
    timeBudgetMs: cliArgs.timeMs,
    maxPlies: cliArgs.maxPlies,
    seed: cliArgs.seed,
    openingRandomPlies: cliArgs.openingRandomPlies,
    epsilon: cliArgs.epsilon,
    augment: cliArgs.augment,
  };

  if (cliArgs.net) {
    // NNUE 自己対戦ブートストラップ: 指定モデルをロードし、探索の評価関数として使う。
    // 重い処理になるため、大規模生成はここでは行わない（疎通確認は極小構成のみ）。
    console.log(`NNUE ネットワークをロード中: ${cliArgs.net}`);
    const net = await loadNetwork(cliArgs.net);
    config.evalFn = (s, p) => net.evaluateState(s, p);
    config.netPath = cliArgs.net;
  }

  const outPath = cliArgs.out;
  const metaPath = metaPathFor(outPath);

  console.log("Ukeja Layer NNUE 教師データ生成");
  console.log(`config: ${JSON.stringify({ ...config, evalFn: undefined })}`);
  console.log(`出力先: ${outPath}`);
  console.log("─".repeat(60));

  const startTime = performance.now();
  const { records, meta } = buildDataset(config);
  const elapsedMs = performance.now() - startTime;

  const outDir = path.dirname(outPath);
  fs.mkdirSync(outDir, { recursive: true });

  const jsonlLines = records.map((r) => JSON.stringify(r)).join("\n") + (records.length > 0 ? "\n" : "");
  fs.writeFileSync(outPath, jsonlLines, "utf8");
  fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), "utf8");

  console.log("");
  console.log("=== 生成結果 ===");
  console.log(`ゲーム数:     ${meta.numGames}`);
  console.log(`元局面数:     ${meta.numPositions}`);
  console.log(`レコード数:   ${meta.numRecords}`);
  console.log(`評価器:       ${meta.generator.evaluator}${meta.generator.net ? ` (${meta.generator.net})` : ""}`);
  console.log(`ラベル分布:   win=${meta.resultDistribution.win} draw=${meta.resultDistribution.draw} loss=${meta.resultDistribution.loss}`);
  console.log(`スコア統計:   min=${meta.scoreStats.min} max=${meta.scoreStats.max} mean=${meta.scoreStats.mean}`);
  console.log(`所要時間:     ${(elapsedMs / 1000).toFixed(2)} 秒`);
  console.log(`書き込み先:   ${outPath}`);
  console.log(`メタ情報:     ${metaPath}`);
}

// ESM のエントリポイント判定（このファイルが直接実行された場合のみ main を呼ぶ）
const isDirectRun = process.argv[1] && (
  process.argv[1].endsWith("genData.mjs")
);
if (isDirectRun) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
