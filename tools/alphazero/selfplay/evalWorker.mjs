#!/usr/bin/env node
/**
 * evalWorker.mjs — P5 自己改善ループの評価ワーカー（並列実行用の1シャード）。
 *
 * 3モード:
 *   --mode vsHeuristic : netA(+MCTS) を共通相手 heuristic(search:dD:Tms) と対戦させ、
 *                        netA視点の {wins, losses, draws} を返す。
 *   --mode headToHead  : netA(+MCTS) vs netB(+MCTS)（直接対決）。
 *   --mode vsSearchNet : netA(+固定sims MCTS) を相手 oppNet+αβ(findBestTurn, NNUE評価) と対戦。
 *                        --oppNet models/gen137.json --sdepth 32 --sms 120。両者の実測ms/turnも集計。
 *
 * MCTS は固定 simulations で決定的に探索する（評価の再現性・速度のため。自己対戦の
 * Dirichlet/温度は OFF）。色は g%2 で交換。結果は JSON 1行で stdout に出す（親が集計）。
 *
 * 使い方:
 *   node evalWorker.mjs --mode vsHeuristic --netA models/genAZ_r1.json \
 *       --games 20 --seed 1 --sims 100 --hdepth 20 --hms 120 --maxPlies 200
 */

import { generateTurns } from "../../../js/ai/moveGen.js";
import { normalizeBoard } from "../../../js/gameLogic.js";
import { findBestTurn } from "../../../js/ai/search.js";
import { WEIGHTS } from "../../../js/ai/evaluate.js";
import { loadNetwork } from "../../../js/ai/nnue/network.js";
import { playTurnMcts } from "../mctsPlayer.mjs";

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const emptyBoard = (size) => Array.from({ length: size }, () => Array.from({ length: size }, () => []));
const nextPlayer = (p) => (p === "white" ? "black" : "white");

/** 2エージェント（chooseTurn を持つ）で1局対戦。勝者色 or null(maxPlies引き分け) を返す。 */
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
  return null;
}

/** 固定 simulations の MCTS エージェント。 */
function mctsAgent(net, sims, rng) {
  return { chooseTurn(state) {
    const r = playTurnMcts(state, net, { simulations: sims, rng });
    return { board: r.board, summonCounts: r.summonCounts };
  } };
}

/** heuristic(search:dD:Tms) エージェント。 */
function heuristicAgent(depth, ms, rng) {
  return { chooseTurn(state) {
    const res = findBestTurn(state, { maxDepth: depth, timeBudgetMs: ms, weights: WEIGHTS, rng });
    return { board: res.turn.board, summonCounts: res.turn.summonCounts };
  } };
}

/** ランダム着手ボット（合法ターンを一様選択）。学習可能性診断のベンチマーク相手。 */
function randomAgent(rng) {
  return { chooseTurn(state) {
    const cands = generateTurns(state);
    const c = cands[Math.floor(rng() * cands.length)];
    return { board: c.board, summonCounts: c.summonCounts };
  } };
}

/** oppNet+αβ(findBestTurn, NNUE評価) エージェント（vsSearchNet の相手側）。 */
function searchNetAgent(net, depth, ms, rng) {
  const evalFn = (s, p) => net.evaluateState(s, p);
  return { chooseTurn(state) {
    const res = findBestTurn(state, { maxDepth: depth, timeBudgetMs: ms, evalFn, rng });
    return { board: res.turn.board, summonCounts: res.turn.summonCounts };
  } };
}

/** 計測ラッパ: chooseTurn の累積時間とターン数を記録する（記録用）。 */
function instrument(agent, stats) {
  return { chooseTurn(state) {
    const t = performance.now();
    const r = agent.chooseTurn(state);
    stats.ms += performance.now() - t;
    stats.turns += 1;
    return r;
  } };
}

async function main() {
  const argv = process.argv.slice(2);
  const get = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
  const mode = get("--mode", "vsHeuristic");
  const netAPath = get("--netA", "models/genAZ001.json");
  const netBPath = get("--netB", null);
  const oppNetPath = get("--oppNet", "models/gen137.json");
  const games = parseInt(get("--games", "20"), 10);
  const seed = parseInt(get("--seed", "1"), 10);
  const sims = parseInt(get("--sims", "100"), 10);
  const hdepth = parseInt(get("--hdepth", "20"), 10);
  const hms = parseInt(get("--hms", "120"), 10);
  const sdepth = parseInt(get("--sdepth", "32"), 10);
  const sms = parseInt(get("--sms", "120"), 10);
  const maxPlies = parseInt(get("--maxPlies", "200"), 10);
  const size = 4;

  const netA = await loadNetwork(netAPath);
  const netB = netBPath ? await loadNetwork(netBPath) : null;
  const oppNet = mode === "vsSearchNet" ? await loadNetwork(oppNetPath) : null;

  const focalStats = { ms: 0, turns: 0 };
  const oppStats = { ms: 0, turns: 0 };

  let wins = 0, losses = 0, draws = 0; // netA(=focal) 視点
  for (let g = 0; g < games; g++) {
    const rng = mulberry32(seed + g * 7919);
    const focal = instrument(mctsAgent(netA, sims, rng), focalStats);
    let oppRaw;
    if (mode === "headToHead") oppRaw = mctsAgent(netB, sims, rng);
    else if (mode === "vsSearchNet") oppRaw = searchNetAgent(oppNet, sdepth, sms, rng);
    else if (mode === "vsRandom") oppRaw = randomAgent(rng);
    else oppRaw = heuristicAgent(hdepth, hms, rng);
    const opp = instrument(oppRaw, oppStats);
    const focalIsWhite = g % 2 === 0; // 色交換
    const winner = focalIsWhite ? playGame(focal, opp, size, maxPlies) : playGame(opp, focal, size, maxPlies);
    const focalColor = focalIsWhite ? "white" : "black";
    if (winner === null) draws++;
    else if (winner === focalColor) wins++;
    else losses++;
  }
  const msTurnFocal = focalStats.turns ? focalStats.ms / focalStats.turns : 0;
  const msTurnOpp = oppStats.turns ? oppStats.ms / oppStats.turns : 0;
  process.stdout.write(JSON.stringify({
    mode, netA: netAPath, netB: netBPath,
    ...(mode === "vsSearchNet" ? { oppNet: oppNetPath, sdepth, sms } : {}),
    games, seed, sims, wins, losses, draws,
    msTurnFocal: Number(msTurnFocal.toFixed(2)), msTurnOpp: Number(msTurnOpp.toFixed(2)),
  }) + "\n");
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
