#!/usr/bin/env node
/**
 * evalGate.mjs — P3 評価ゲート: v0(genAZ001)+MCTS vs gen137+αβ を
 * 同一 ms/ターンで色交換つきに対戦させ、AZ視点の勝率と Wilson 95%CI を報告する。
 *
 * 公平性: 両者ともターン(=1ply)あたり同一 timeBudgetMs。実測 ms/turn も出力する。
 *
 * 使い方:
 *   node tools/alphazero/evalGate.mjs [--games N] [--ms 120] [--seed 1] [--maxPlies 200]
 */

import { generateTurns } from "../../js/ai/moveGen.js";
import { normalizeBoard } from "../../js/gameLogic.js";
import { findBestTurn } from "../../js/ai/search.js";
import { loadNetwork } from "../../js/ai/nnue/network.js";
import { playTurnMcts } from "./mctsPlayer.mjs";

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
const nextPlayer = (p) => (p === "white" ? "black" : "white");

function wilson(wins, n, z = 1.96) {
  if (n === 0) return [0, 0, 0];
  const p = wins / n;
  const denom = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [p, center - half, center + half];
}

function parseArgs(argv) {
  const a = { games: 40, ms: 120, seed: 1, maxPlies: 200, net: "models/genAZ001.json" };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--games") a.games = parseInt(argv[++i], 10);
    else if (argv[i] === "--ms") a.ms = parseInt(argv[++i], 10);
    else if (argv[i] === "--seed") a.seed = parseInt(argv[++i], 10);
    else if (argv[i] === "--maxPlies") a.maxPlies = parseInt(argv[++i], 10);
    else if (argv[i] === "--net") a.net = argv[++i];
  }
  return a;
}

/** 計測ラッパ: chooseTurn の累積時間とターン数を記録する。 */
function instrument(agent, stats) {
  return {
    name: agent.name,
    chooseTurn(state) {
      const t0 = performance.now();
      const r = agent.chooseTurn(state);
      stats.ms += performance.now() - t0;
      stats.turns += 1;
      return r;
    },
  };
}

function playGame(whiteAgent, blackAgent, size, maxPlies) {
  let board = normalizeBoard(emptyBoard(size), size);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";
  let ply = 0;
  while (ply < maxPlies) {
    const state = { board: normalizeBoard(board, size), summonCounts: { ...summonCounts }, currentPlayer, boardSize: size };
    if (generateTurns(state).length === 0) return nextPlayer(currentPlayer);
    const chosen = (currentPlayer === "white" ? whiteAgent : blackAgent).chooseTurn(state);
    board = normalizeBoard(chosen.board, size);
    summonCounts = chosen.summonCounts;
    currentPlayer = nextPlayer(currentPlayer);
    ply++;
  }
  return null; // maxPlies 到達（引き分け扱い）
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const azNet = await loadNetwork(args.net);
  const gen137 = await loadNetwork("models/gen137.json");
  const evalFn = (s, p) => gen137.evaluateState(s, p);

  const azStats = { ms: 0, turns: 0 };
  const abStats = { ms: 0, turns: 0 };

  let azWins = 0, abWins = 0, draws = 0;
  const size = 4;

  // ── 公平化（CRITICAL2）────────────────────────────────────────────────
  // findBestTurn（既存js、変更不可）は反復深化がイテレーション境界でしか時間を見ないため
  // timeBudgetMs を超過する（実測で ~1.7倍）。az(MCTS) は deadline 厳守で予算ちょうど。
  // これでは「同一予算」でも実効思考時間が az に不利。
  // → az の予算を「αβ の実測 ms/turn の走行平均」に動的に合わせ、実効思考時間を揃える。
  //   初期ターンは --ms を種にし、対局が進むと αβ 実測へ収束する。
  // また gen137+αβ は「時間律速の反復深化（人為的 depth 上限なし）」で運用する
  //   ＝ gen137 が vs heuristic 95.6% を出した本来の構成。深さは時間で自然に律速される。
  const AB_MAX_DEPTH = 32; // 事実上の上限なし（4x4 で 32手先まで読むことはない）

  console.log(`評価ゲート: az(genAZ001)+MCTS vs gen137+αβ  games=${args.games} 基準ms=${args.ms}`);
  console.log(`公平化: az予算 = αβ実測ms/turn の走行平均（実効思考時間を一致）／αβ=時間律速(depth上限${AB_MAX_DEPTH})`);

  for (let g = 0; g < args.games; g++) {
    const rng = mulberry32(args.seed + g * 7919);

    const azAgent = instrument(
      { name: "az", chooseTurn(state) {
        // αβ の走行平均実測 ms を az の予算にする（データが無い初期は --ms）
        const budget = abStats.turns > 0 ? Math.max(1, Math.round(abStats.ms / abStats.turns)) : args.ms;
        const r = playTurnMcts(state, azNet, { timeBudgetMs: budget, rng });
        return { board: r.board, summonCounts: r.summonCounts };
      } },
      azStats
    );
    const abAgent = instrument(
      { name: "ab", chooseTurn(state) {
        const res = findBestTurn(state, { maxDepth: AB_MAX_DEPTH, timeBudgetMs: args.ms, evalFn, rng });
        return { board: res.turn.board, summonCounts: res.turn.summonCounts };
      } },
      abStats
    );

    const azIsWhite = g % 2 === 0; // 色交換
    const winner = azIsWhite ? playGame(azAgent, abAgent, size, args.maxPlies) : playGame(abAgent, azAgent, size, args.maxPlies);
    const azColor = azIsWhite ? "white" : "black";
    if (winner === null) draws++;
    else if (winner === azColor) azWins++;
    else abWins++;

    if ((g + 1) % 5 === 0 || g === args.games - 1) {
      const [p, lo, hi] = wilson(azWins, azWins + abWins);
      const azMsAvg = azStats.turns ? azStats.ms / azStats.turns : 0;
      const abMsAvg = abStats.turns ? abStats.ms / abStats.turns : 0;
      process.stdout.write(
        `  [${g + 1}/${args.games}] az=${azWins} ab=${abWins} draw=${draws} ` +
          `winrate=${(p * 100).toFixed(1)}% CI[${(lo * 100).toFixed(1)},${(hi * 100).toFixed(1)}] ` +
          `ms/turn az=${azMsAvg.toFixed(0)} ab=${abMsAvg.toFixed(0)}\n`
      );
    }
  }

  const decisive = azWins + abWins;
  const [p, lo, hi] = wilson(azWins, decisive);
  const azMsAvg = azStats.ms / azStats.turns;
  const abMsAvg = abStats.ms / abStats.turns;
  const timeSkew = Math.abs(azMsAvg - abMsAvg) / abMsAvg;
  console.log("─".repeat(60));
  console.log(`AZ+MCTS 勝率(決着のみ): ${azWins}/${decisive} = ${(p * 100).toFixed(1)}%  Wilson95%CI [${(lo * 100).toFixed(1)}%, ${(hi * 100).toFixed(1)}%]`);
  console.log(`引き分け(maxPlies): ${draws}`);
  console.log(`実測 ms/turn: az=${azMsAvg.toFixed(1)} (${azStats.turns}t)  ab=${abMsAvg.toFixed(1)} (${abStats.turns}t)  時間差=${(timeSkew * 100).toFixed(1)}%`);
  if (timeSkew > 0.2) console.log(`  ⚠ 実効思考時間差が ±20% を超過（公平性に注意して解釈すること）`);
  console.log(`判定: ${p >= 0.5 ? "互角以上（P3完了条件クリア）" : "未達"}`);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
