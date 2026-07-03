#!/usr/bin/env node
/**
 * selfplay.mjs — 自己対戦スクリプト
 *
 * 使い方:
 *   node tools/selfplay.mjs [--games N] [--size N] [--seed N] [--maxPlies N]
 *                           [--white <type>] [--black <type>] [--swap]
 *
 * プレイヤータイプ: random, greedy, search, search:d3, search:d4:500
 *   search:d<depth>:<timeBudgetMs> の形式でパラメータ指定可能。
 *   depth のみ省略した場合: search はデフォルト (d4:1000ms)。
 * デフォルト: games=50, size=4, maxPlies=400, seed=42, white=random, black=random
 *
 * --swap を指定すると、ゲームの前半はそのままのカラー割り当て、
 * 後半は white/black を入れ替えてプレイし、エージェントタイプ別の勝率を報告する。
 */

import { generateTurns } from "../js/ai/moveGen.js";
import { checkWinCondition, maxSummonsFor, normalizeBoard } from "../js/gameLogic.js";
import { randomPlayer, greedyPlayer, searchPlayer } from "../js/ai/players.js";

// ───────────────────────── CLI パース ─────────────────────────
function parseArgs(argv) {
  const args = {
    games: 50,
    size: 4,
    seed: 42,
    maxPlies: 400,
    white: "random",
    black: "random",
    swap: false,
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--games")   args.games   = parseInt(argv[i + 1], 10);
    if (argv[i] === "--size")    args.size    = parseInt(argv[i + 1], 10);
    if (argv[i] === "--seed")    args.seed    = parseInt(argv[i + 1], 10);
    if (argv[i] === "--maxPlies") args.maxPlies = parseInt(argv[i + 1], 10);
    if (argv[i] === "--white")   args.white   = argv[i + 1];
    if (argv[i] === "--black")   args.black   = argv[i + 1];
    if (argv[i] === "--swap")    args.swap    = true;
  }
  return args;
}

// ───────────────────────── mulberry32 PRNG ─────────────────────────
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ───────────────────────── プレイヤー生成 ─────────────────────────

/**
 * タイプ名からプレイヤーファクトリを返す。
 * search:d<depth>:<timeBudgetMs> の形式もサポートする。
 *
 * @param {string} type
 * @param {() => number} rng
 * @returns {{ name: string, chooseTurn(state): object|null }}
 */
function makePlayer(type, rng) {
  if (type === "greedy")  return greedyPlayer(rng);
  if (type === "random")  return randomPlayer(rng);

  // search, search:d3, search:d4:500 などをパース
  if (type === "search" || type.startsWith("search:")) {
    const parts = type.split(":");
    // parts[0] === "search"
    // parts[1] (optional) === "d<depth>"
    // parts[2] (optional) === "<timeBudgetMs>"
    let maxDepth = 4;
    let timeBudgetMs = 1000;

    if (parts[1]) {
      const depthStr = parts[1];
      if (depthStr.startsWith("d")) {
        maxDepth = parseInt(depthStr.slice(1), 10);
        if (isNaN(maxDepth)) throw new Error(`Invalid depth in "${type}"`);
      } else {
        throw new Error(`search の depth は "d<N>" 形式で指定してください: "${type}"`);
      }
    }
    if (parts[2]) {
      timeBudgetMs = parseInt(parts[2], 10);
      if (isNaN(timeBudgetMs)) throw new Error(`Invalid timeBudgetMs in "${type}"`);
    }

    return searchPlayer(rng, { maxDepth, timeBudgetMs });
  }

  throw new Error(`Unknown player type: ${type}. Valid types: random, greedy, search, search:d3, search:d4:500`);
}

// ───────────────────────── ゲームループ ─────────────────────────
function emptyBoard(size) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => []));
}

function nextPlayer(p) {
  return p === "white" ? "black" : "white";
}

/**
 * 1ゲームを実行して結果を返す。
 * @param {number} boardSize
 * @param {number} maxPlies
 * @param {{ name: string, chooseTurn(state): object|null, lastSearchInfo?: object }} whiteAgent
 * @param {{ name: string, chooseTurn(state): object|null, lastSearchInfo?: object }} blackAgent
 * @returns {{ winner: string|null, winnerAgentType: string|null,
 *             plies: number, aborted: boolean,
 *             totalCandidates: number, maxBranch: number,
 *             totalGenMs: number, errors: string[],
 *             agentChooseMs: { [agentType: string]: { totalMs: number, count: number } },
 *             searchStats: { [agentType: string]: { totalDepth: number, totalNodes: number, count: number } } }}
 */
function playGame(boardSize, maxPlies, whiteAgent, blackAgent) {
  const maxSummons = maxSummonsFor(boardSize);
  let board = emptyBoard(boardSize);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";
  let plies = 0;
  let totalCandidates = 0;
  let maxBranch = 0;
  let totalGenMs = 0;
  const errors = [];
  // エージェントタイプ別の chooseTurn 時間集計
  const agentChooseMs = {};
  // サーチエージェントの探索統計（深さ・ノード数）
  const searchStats = {};

  const agents = { white: whiteAgent, black: blackAgent };

  while (plies < maxPlies) {
    const normBoard = normalizeBoard(board, boardSize);
    const state = { board: normBoard, summonCounts: { ...summonCounts }, currentPlayer, boardSize };

    // generateTurns で分岐数計測（整合性チェック用）
    const t0 = performance.now();
    const turns = generateTurns(state);
    const dt = performance.now() - t0;
    totalGenMs += dt;

    const branch = turns.length;
    totalCandidates += branch;
    if (branch > maxBranch) maxBranch = branch;

    // generateTurns が空 → 現在のプレイヤーは手なし → 相手の勝ち
    if (branch === 0) {
      const winner = nextPlayer(currentPlayer);
      const winnerAgentType = agents[winner].name;
      // checkWinCondition による整合性チェック
      const meta = { maxSummons, boardSize };
      const win = checkWinCondition({ board: normBoard, summonCounts, currentPlayer }, meta);
      if (!win) {
        errors.push(`ply=${plies} player=${currentPlayer}: generateTurns empty but checkWinCondition=false`);
      }
      return { winner, winnerAgentType, plies, aborted: false, totalCandidates, maxBranch, totalGenMs, errors, agentChooseMs, searchStats };
    }

    // checkWinCondition が true なのに手があればミスマッチ
    {
      const meta = { maxSummons, boardSize };
      const win = checkWinCondition({ board: normBoard, summonCounts, currentPlayer }, meta);
      if (win) {
        errors.push(`ply=${plies} player=${currentPlayer}: checkWinCondition=true but generateTurns returned ${branch} candidates`);
      }
    }

    // エージェントの chooseTurn を計測
    const agent = agents[currentPlayer];
    const tc0 = performance.now();
    const chosen = agent.chooseTurn(state);
    const tcDt = performance.now() - tc0;

    // エージェントタイプ別に時間を集計
    if (!agentChooseMs[agent.name]) {
      agentChooseMs[agent.name] = { totalMs: 0, count: 0 };
    }
    agentChooseMs[agent.name].totalMs += tcDt;
    agentChooseMs[agent.name].count++;

    // サーチエージェントの探索統計を集計
    if (agent.lastSearchInfo) {
      const si = agent.lastSearchInfo;
      if (!searchStats[agent.name]) {
        searchStats[agent.name] = { totalDepth: 0, totalNodes: 0, count: 0 };
      }
      searchStats[agent.name].totalDepth += si.depthReached;
      searchStats[agent.name].totalNodes += si.nodes;
      searchStats[agent.name].count++;
    }

    if (!chosen) {
      // chooseTurn が null を返した（手なし）→ 相手の勝ち
      const winner = nextPlayer(currentPlayer);
      const winnerAgentType = agents[winner].name;
      return { winner, winnerAgentType, plies, aborted: false, totalCandidates, maxBranch, totalGenMs, errors, agentChooseMs, searchStats };
    }

    // 状態更新
    board = normalizeBoard(chosen.board, boardSize);
    summonCounts = chosen.summonCounts;
    currentPlayer = nextPlayer(currentPlayer);
    plies++;
  }

  // maxPlies に達した → 中断
  return { winner: null, winnerAgentType: null, plies, aborted: true, totalCandidates, maxBranch, totalGenMs, errors, agentChooseMs, searchStats };
}

// ───────────────────────── メイン ─────────────────────────
function main() {
  const args = parseArgs(process.argv.slice(2));
  const { games, size, seed, maxPlies, swap } = args;
  const whiteType = args.white;
  const blackType = args.black;

  console.log(`Ukeja Layer 自己対戦 — games=${games} size=${size} seed=${seed} maxPlies=${maxPlies}`);
  console.log(`プレイヤー: white=${whiteType}  black=${blackType}${swap ? "  [--swap 有効]" : ""}`);
  console.log("─".repeat(60));

  // ゲームごとにシードを派生させて再現性を確保
  let masterRng = mulberry32(seed);

  const stats = {
    whiteWins: 0,
    blackWins: 0,
    aborted: 0,
    totalPlies: 0,
    totalCandidates: 0,
    maxBranch: 0,
    totalGenMs: 0,
    totalErrors: 0,
  };

  // エージェントタイプ別の勝利数（swap 時に使用）
  const agentWins = {};
  const agentGames = {};   // 決着ゲーム数（引き分け除く）

  // エージェントタイプ別の chooseTurn 時間集計
  const agentChooseMsTotal = {};
  // サーチエージェントの探索統計合計
  const searchStatsTotal = {};

  // swap 時: 前半は通常配色、後半は入れ替え配色
  const halfPoint = swap ? Math.floor(games / 2) : games;

  for (let g = 0; g < games; g++) {
    // ゲームごとに独立した PRNG を作る
    const gameSeed = (masterRng() * 0xffffffff) >>> 0;
    const agentRng  = mulberry32(gameSeed);

    // swap 後半はカラー割り当てを反転
    const isSwapped = swap && g >= halfPoint;
    const wType = isSwapped ? blackType : whiteType;
    const bType = isSwapped ? whiteType : blackType;

    // エージェント PRNG を分割（再現性のため）
    const wRng = mulberry32((agentRng() * 0xffffffff) >>> 0);
    const bRng = mulberry32((agentRng() * 0xffffffff) >>> 0);

    const wAgent = makePlayer(wType, wRng);
    const bAgent = makePlayer(bType, bRng);

    const result = playGame(size, maxPlies, wAgent, bAgent);

    if (result.winner === "white") stats.whiteWins++;
    else if (result.winner === "black") stats.blackWins++;
    if (result.aborted) stats.aborted++;

    stats.totalPlies += result.plies;
    stats.totalCandidates += result.totalCandidates;
    if (result.maxBranch > stats.maxBranch) stats.maxBranch = result.maxBranch;
    stats.totalGenMs += result.totalGenMs;
    stats.totalErrors += result.errors.length;

    // エージェントタイプ別の勝利数を集計
    if (result.winner && !result.aborted) {
      const winnerColor = result.winner; // "white" | "black"
      // swap 時は実際に white/black に割り当てられたエージェントタイプを参照
      const winnerType = winnerColor === "white" ? wType : bType;
      agentWins[winnerType] = (agentWins[winnerType] || 0) + 1;
    }
    // 決着ゲーム数
    if (!result.aborted && result.winner) {
      // 両エージェントのゲーム参加数をカウント
      agentGames[wType] = (agentGames[wType] || 0) + 1;
      if (wType !== bType) {
        agentGames[bType] = (agentGames[bType] || 0) + 1;
      }
    }

    // エージェントタイプ別 chooseTurn 時間集計
    for (const [aType, ms] of Object.entries(result.agentChooseMs)) {
      if (!agentChooseMsTotal[aType]) {
        agentChooseMsTotal[aType] = { totalMs: 0, count: 0 };
      }
      agentChooseMsTotal[aType].totalMs += ms.totalMs;
      agentChooseMsTotal[aType].count   += ms.count;
    }

    // サーチエージェントの探索統計集計
    for (const [aType, ss] of Object.entries(result.searchStats)) {
      if (!searchStatsTotal[aType]) {
        searchStatsTotal[aType] = { totalDepth: 0, totalNodes: 0, count: 0 };
      }
      searchStatsTotal[aType].totalDepth += ss.totalDepth;
      searchStatsTotal[aType].totalNodes += ss.totalNodes;
      searchStatsTotal[aType].count      += ss.count;
    }

    if (result.errors.length > 0) {
      for (const e of result.errors) {
        console.error(`  [ERROR] game=${g + 1}: ${e}`);
      }
    }
  }

  const totalPly = stats.totalPlies;
  const avgGameLength = (totalPly / games).toFixed(1);
  const avgBranch = totalPly > 0 ? (stats.totalCandidates / totalPly).toFixed(1) : "N/A";
  const avgGenMs = totalPly > 0 ? (stats.totalGenMs / totalPly).toFixed(3) : "N/A";

  console.log("");
  console.log("=== 結果 ===");
  console.log(`白 勝利:         ${stats.whiteWins}`);
  console.log(`黒 勝利:         ${stats.blackWins}`);
  console.log(`中断 (maxPlies): ${stats.aborted}`);
  console.log(`平均ゲーム長:    ${avgGameLength} plies`);
  console.log(`平均分岐数:      ${avgBranch}`);
  console.log(`最大分岐数:      ${stats.maxBranch}`);
  console.log(`generateTurns 平均時間: ${avgGenMs} ms/ply`);
  console.log(`エラー件数:      ${stats.totalErrors}`);

  // エージェントタイプ別の勝利数（swap 時のみ意味あり）
  if (swap || whiteType !== blackType) {
    console.log("");
    console.log("=== エージェントタイプ別勝率 ===");
    const allTypes = new Set([whiteType, blackType]);
    for (const aType of allTypes) {
      const wins = agentWins[aType] || 0;
      const decided = agentGames[aType] || 0;
      const rate = decided > 0 ? ((wins / decided) * 100).toFixed(1) : "N/A";
      console.log(`  ${aType}: ${wins}勝 / ${decided}決着 (勝率 ${rate}%)`);
    }
  }

  // エージェントタイプ別の chooseTurn 平均時間
  console.log("");
  console.log("=== chooseTurn 平均時間 ===");
  for (const [aType, ms] of Object.entries(agentChooseMsTotal)) {
    const avg = ms.count > 0 ? (ms.totalMs / ms.count).toFixed(3) : "N/A";
    console.log(`  ${aType}: ${avg} ms/ply (${ms.count} plies)`);
  }

  // サーチエージェントの探索統計
  if (Object.keys(searchStatsTotal).length > 0) {
    console.log("");
    console.log("=== サーチエージェント探索統計 ===");
    for (const [aType, ss] of Object.entries(searchStatsTotal)) {
      const avgDepth = ss.count > 0 ? (ss.totalDepth / ss.count).toFixed(2) : "N/A";
      const avgNodes = ss.count > 0 ? Math.round(ss.totalNodes / ss.count) : "N/A";
      console.log(`  ${aType}: 平均到達深さ=${avgDepth}  平均ノード数=${avgNodes}  (${ss.count} plies)`);
    }
  }

  if (stats.totalErrors > 0) {
    process.exit(1);
  }
}

main();
