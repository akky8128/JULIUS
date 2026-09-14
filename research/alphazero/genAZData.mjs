#!/usr/bin/env node
/**
 * genAZData.mjs — AlphaZero移行 P2: 蒸留ブートストラップ用教師データ生成器。
 *
 * research/nnue/genData.mjs の構造（mulberry32 / buildDataset純関数とmainの分離 /
 * シャード用 --seed/--out / --net でNNUE evalFn）を踏襲しつつ、
 * 「マイクロステップごとに1レコード」の policy/value 教師を出力する。
 *
 * 毎ply:
 *   - findBestTurn を呼び、policy教師 = searchResult.turn（ε-ランダムで実際の
 *     着手が別でも教師は常に探索選択手）
 *   - decomposeTurn でマイクロアクション列化し、各マイクロステップの
 *     「採用直前の状態」から1レコードを作る:
 *       { pid, ply, microStep, indices, scalars, turnContext(26),
 *         policyTarget(0..80), score, result, sym, size, pov }
 *   - score/result はターン単位の値を全マイクロステップに複製
 *     （value 意味論 =「このマイクロ状態から手番プレイヤーが勝つ確率」。
 *      ターン内は手番不変なのでこれで正しい）
 *
 * D4 augment=8:
 *   indices は transformState 経由（既存どおり）、turnContext は
 *   transformTurnContext、policyTarget は transformActionIndex で共変変換。
 *   scalars/score/result は sym 不変。
 *
 * 使い方:
 *   node research/alphazero/genAZData.mjs [--games N] [--depth N] [--timeMs N]
 *     [--maxPlies N] [--seed N] [--pidOffset N] [--openingRandomPlies N]
 *     [--epsilon N] [--augment 1|8] [--net research/models/gen137.json] [--out <path>]
 */

import fs from "node:fs";
import path from "node:path";

import { generateTurns } from "../../core/ai/moveGen.js";
import { findBestTurn } from "../../core/ai/search.js";
import { WEIGHTS } from "../../core/ai/evaluate.js";
import { loadNetwork } from "../../core/nnue/network.js";
import { normalizeBoard } from "../../core/gameLogic.js";
import {
  NUM_SYMMETRIES,
  extractFeatures,
  transformState,
} from "../../core/nnue/features.js";
import { decomposeTurn } from "./turnToMicro.mjs";
import {
  TURN_CONTEXT_DIM,
  extractTurnContext,
  transformTurnContext,
  transformActionIndex,
} from "../../core/az/azFeatures.mjs";

// ───────────────────────── mulberry32 PRNG（genData.mjs と同一実装） ─────────────────────────

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function emptyBoard(size) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => []));
}

function nextPlayer(p) {
  return p === "white" ? "black" : "white";
}

// ───────────────────────── 1ゲーム分の対局・記録 ─────────────────────────

/**
 * 1ゲームを対戦し、ターン単位の一時レコード（分解済みマイクロ列付き）を返す。
 * @returns {{ tempTurns: Array<object>, winner: string|null, plies: number }}
 */
function playAndRecordGame(config, rng, searchRng) {
  const { size, maxDepth, timeBudgetMs, maxPlies, openingRandomPlies, epsilon, evalFn } = config;

  let board = emptyBoard(size);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";
  let ply = 0;

  const tempTurns = [];
  let winner = null;

  while (ply < maxPlies) {
    const state = {
      board: normalizeBoard(board, size),
      summonCounts: { ...summonCounts },
      currentPlayer,
      boardSize: size,
    };

    const candidates = generateTurns(state);
    if (candidates.length === 0) {
      winner = nextPlayer(currentPlayer);
      break;
    }

    const searchOptions = evalFn
      ? { maxDepth, timeBudgetMs, evalFn, rng: searchRng }
      : { maxDepth, timeBudgetMs, weights: WEIGHTS, rng: searchRng };
    const searchResult = findBestTurn(state, searchOptions);
    const teacherTurn = searchResult.turn || candidates[Math.floor(rng() * candidates.length)];

    // policy教師 = 探索選択手をマイクロ列に分解（違反があればここで throw = 握りつぶさない）
    const { actionIndices, contexts } = decomposeTurn(state, teacherTurn);

    tempTurns.push({
      ply,
      mover: currentPlayer,
      score: searchResult.score,
      actionIndices,
      contexts,
    });

    // 着手選択: openingRandomPlies 中 or ε でランダム手（教師とは独立）
    const useRandom = ply < openingRandomPlies || rng() < epsilon;
    const chosen = useRandom
      ? candidates[Math.floor(rng() * candidates.length)]
      : teacherTurn;

    board = normalizeBoard(chosen.board, size);
    summonCounts = chosen.summonCounts;
    currentPlayer = nextPlayer(currentPlayer);
    ply++;
  }

  return { tempTurns, winner, plies: ply };
}

// ───────────────────────── D4展開してレコード化 ─────────────────────────

/**
 * 1ターン分（複数マイクロステップ）を D4 展開して records へ push する。
 * pid はターン内の全マイクロステップ・全 sym で共通（グループ分割用）。
 */
function expandTurnRecords(tempTurn, result, augment, pid, size, outRecords) {
  const { ply, mover, score, actionIndices, contexts } = tempTurn;
  const symCount = augment === 8 ? NUM_SYMMETRIES : 1;

  for (let step = 0; step < actionIndices.length; step++) {
    const ctx = contexts[step];
    const microState = {
      board: ctx.board,
      summonCounts: ctx.summonCounts,
      currentPlayer: mover,
      boardSize: size,
    };
    const turnContext0 = extractTurnContext(ctx);
    const { scalars } = extractFeatures(microState, mover); // scalars は sym 不変

    for (let sym = 0; sym < symCount; sym++) {
      const symState = sym === 0 ? microState : transformState(microState, sym);
      const { indices } = extractFeatures(symState, mover);
      const turnContext =
        sym === 0 ? Array.from(turnContext0) : Array.from(transformTurnContext(turnContext0, sym, size));
      const policyTarget = transformActionIndex(actionIndices[step], sym, size);

      outRecords.push({
        pid,
        size,
        pov: mover,
        ply,
        microStep: step,
        sym,
        indices,
        scalars,
        turnContext,
        policyTarget,
        score,
        result,
      });
    }
  }
}

// ───────────────────────── buildDataset（純粋関数） ─────────────────────────

export function buildDataset(config) {
  const {
    games, size, maxDepth, timeBudgetMs, maxPlies, seed,
    openingRandomPlies, epsilon, augment, evalFn, pidOffset = 0,
  } = config;

  const masterRng = mulberry32(seed);
  const records = [];
  let nextPid = pidOffset;

  let winResults = 0, drawResults = 0, lossResults = 0;
  let turnsTotal = 0, microStepsTotal = 0;
  const policyKind = { move: 0, summonOrEliminate: 0, end: 0 };

  for (let g = 0; g < games; g++) {
    const gameSeed = (masterRng() * 0xffffffff) >>> 0;
    const gameRng = mulberry32(gameSeed);
    const moveRng = mulberry32((gameRng() * 0xffffffff) >>> 0);
    const searchRng = mulberry32((gameRng() * 0xffffffff) >>> 0);

    const { tempTurns, winner } = playAndRecordGame(
      { size, maxDepth, timeBudgetMs, maxPlies, openingRandomPlies, epsilon, evalFn },
      moveRng,
      searchRng
    );

    for (const tempTurn of tempTurns) {
      let result;
      if (winner === null) { result = 0.5; drawResults++; }
      else if (winner === tempTurn.mover) { result = 1; winResults++; }
      else { result = 0; lossResults++; }

      const pid = nextPid++;
      expandTurnRecords(tempTurn, result, augment, pid, size, records);
      turnsTotal++;
      microStepsTotal += tempTurn.actionIndices.length;
      for (const a of tempTurn.actionIndices) {
        if (a === 80) policyKind.end++;
        else if (a >= 64) policyKind.summonOrEliminate++;
        else policyKind.move++;
      }
    }
  }

  const meta = {
    generatedAt: new Date().toISOString(),
    format: "az-distill-v1",
    turnContextDim: TURN_CONTEXT_DIM,
    config: { ...config, evalFn: undefined },
    numGames: games,
    numTurns: turnsTotal,
    numMicroSteps: microStepsTotal,
    numRecords: records.length,
    numPositions: nextPid - pidOffset,
    pidRange: [pidOffset, nextPid],
    generator: { evaluator: evalFn ? "nnue" : "heuristic", net: config.netPath || null },
    resultDistribution: { win: winResults, draw: drawResults, loss: lossResults },
    policyTargetKinds: policyKind,
  };

  return { records, meta };
}

// ───────────────────────── CLI ─────────────────────────

function parseArgs(argv) {
  const args = {
    games: 20,
    size: 4,
    depth: 3,
    timeMs: 120,
    maxPlies: 200,
    seed: 1,
    pidOffset: 0,
    openingRandomPlies: 12,
    epsilon: 0.25,
    augment: 8,
    out: "research/alphazero/data/az000.jsonl",
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
    else if (key === "--pidOffset") args.pidOffset = parseInt(val, 10);
    else if (key === "--openingRandomPlies") args.openingRandomPlies = parseInt(val, 10);
    else if (key === "--epsilon") args.epsilon = parseFloat(val);
    else if (key === "--augment") args.augment = parseInt(val, 10);
    else if (key === "--out") args.out = val;
    else if (key === "--net") args.net = val;
  }
  return args;
}

function metaPathFor(outPath) {
  const ext = path.extname(outPath);
  const base = ext ? outPath.slice(0, -ext.length) : outPath;
  return `${base}.meta.json`;
}

async function main() {
  const cliArgs = parseArgs(process.argv.slice(2));
  const config = {
    games: cliArgs.games,
    size: cliArgs.size,
    maxDepth: cliArgs.depth,
    timeBudgetMs: cliArgs.timeMs,
    maxPlies: cliArgs.maxPlies,
    seed: cliArgs.seed,
    pidOffset: cliArgs.pidOffset,
    openingRandomPlies: cliArgs.openingRandomPlies,
    epsilon: cliArgs.epsilon,
    augment: cliArgs.augment,
  };

  if (cliArgs.net) {
    console.log(`NNUE ネットワークをロード中: ${cliArgs.net}`);
    const net = await loadNetwork(cliArgs.net);
    config.evalFn = (s, p) => net.evaluateState(s, p);
    config.netPath = cliArgs.net;
  }

  const outPath = cliArgs.out;
  console.log("JuliUs AlphaZero 蒸留教師データ生成 (P2)");
  console.log(`config: ${JSON.stringify({ ...config, evalFn: undefined })}`);

  const startTime = performance.now();
  const { records, meta } = buildDataset(config);
  const elapsedMs = performance.now() - startTime;

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const jsonlLines = records.map((r) => JSON.stringify(r)).join("\n") + (records.length > 0 ? "\n" : "");
  fs.writeFileSync(outPath, jsonlLines, "utf8");
  fs.writeFileSync(metaPathFor(outPath), JSON.stringify(meta, null, 2), "utf8");

  console.log("=== 生成結果 ===");
  console.log(`ゲーム数: ${meta.numGames} / ターン数: ${meta.numTurns} / マイクロ: ${meta.numMicroSteps}`);
  console.log(`レコード数: ${meta.numRecords} (augment=${config.augment})`);
  console.log(`policy種別: ${JSON.stringify(meta.policyTargetKinds)}`);
  console.log(`結果分布: ${JSON.stringify(meta.resultDistribution)}`);
  console.log(`所要時間: ${(elapsedMs / 1000).toFixed(1)} 秒 → ${outPath}`);
}

const isDirectRun = process.argv[1] && process.argv[1].endsWith("genAZData.mjs");
if (isDirectRun) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
