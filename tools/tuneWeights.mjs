#!/usr/bin/env node
/**
 * tuneWeights.mjs — 評価関数ウェイトのチューニングスクリプト
 *
 * 使い方:
 *   node tools/tuneWeights.mjs [--depth 2] [--timeMs 300] [--games 100]
 *                              [--size 4] [--seed 1] [--maxPlies 400]
 *                              [--a <name>] [--b <name>]
 *
 * デフォルト（引数なし）: 全候補を baseline と対戦させてサマリーテーブルを表示する。
 * --a と --b を両方指定すると単一マッチを実行する。
 *
 * 注意: depth=3 だと 100ゲームで 5〜15 分かかる場合があるため、
 *       デフォルトは depth=2, timeMs=300 に設定しています。
 *       統計的に意味のある結果には --games 100 以上を推奨。
 */

import { generateTurns } from "../js/ai/moveGen.js";
import { checkWinCondition, maxSummonsFor, normalizeBoard } from "../js/gameLogic.js";
import { searchPlayer } from "../js/ai/players.js";

// ───────────────────────── 候補ウェイト定義 ─────────────────────────

/**
 * テストする候補ウェイト一覧。
 * キーが候補名、値がウェイトオブジェクト。
 */
export const CANDIDATES = {
  // 現在のデフォルト値（ベースライン）
  baseline: { top: 10, buried: 3, material: 4, elimChance: 2, mobility: 1 },

  // material を引き上げ（仮説: material は過小評価されている）
  moreMaterial: { top: 10, buried: 3, material: 7, elimChance: 2, mobility: 1 },

  // mobility を引き上げ（仮説: mobility は過小評価されている）
  moreMobility: { top: 10, buried: 3, material: 4, elimChance: 2, mobility: 3 },

  // material と mobility の両方を引き上げ（仮説を両方テスト）
  moreBoth: { top: 10, buried: 3, material: 7, elimChance: 2, mobility: 3 },

  // より積極的に両方引き上げ
  moreBothStrong: { top: 10, buried: 3, material: 8, elimChance: 2, mobility: 4 },

  // elimChance も一緒に引き上げ（サイドエフェクト検証）
  moreAllTactical: { top: 10, buried: 3, material: 7, elimChance: 4, mobility: 3 },

  // top をやや下げて他を相対的に引き上げ（別アプローチ）
  rebalanced: { top: 8, buried: 3, material: 6, elimChance: 3, mobility: 2 },
};

// ───────────────────────── CLI パース ─────────────────────────

function parseArgs(argv) {
  const args = {
    depth: 2,        // デフォルト depth=2（depth=3 は速度が遅い）
    timeMs: 300,     // 1手あたりのタイムバジェット (ms)
    games: 100,      // マッチあたりのゲーム数
    size: 4,         // ボードサイズ
    seed: 1,         // ベースシード
    maxPlies: 400,   // 1ゲームあたりの最大手数
    a: null,         // 単一マッチ用: 候補A
    b: null,         // 単一マッチ用: 候補B
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--depth")   args.depth   = parseInt(argv[i + 1], 10);
    if (argv[i] === "--timeMs")  args.timeMs  = parseInt(argv[i + 1], 10);
    if (argv[i] === "--games")   args.games   = parseInt(argv[i + 1], 10);
    if (argv[i] === "--size")    args.size    = parseInt(argv[i + 1], 10);
    if (argv[i] === "--seed")    args.seed    = parseInt(argv[i + 1], 10);
    if (argv[i] === "--maxPlies") args.maxPlies = parseInt(argv[i + 1], 10);
    if (argv[i] === "--a")       args.a       = argv[i + 1];
    if (argv[i] === "--b")       args.b       = argv[i + 1];
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

// ───────────────────────── ゲームループ ─────────────────────────

function emptyBoard(size) {
  return Array.from({ length: size }, () =>
    Array.from({ length: size }, () => [])
  );
}

function nextPlayer(p) {
  return p === "white" ? "black" : "white";
}

/**
 * 1ゲームを実行して結果を返す。
 * selfplay.mjs の playGame() に相当するが、軽量版。
 *
 * @param {number} boardSize
 * @param {number} maxPlies
 * @param {{ chooseTurn(state): object|null }} whiteAgent
 * @param {{ chooseTurn(state): object|null }} blackAgent
 * @returns {{ winner: "white"|"black"|null, aborted: boolean, plies: number }}
 */
function playGame(boardSize, maxPlies, whiteAgent, blackAgent) {
  const maxSummons = maxSummonsFor(boardSize);
  let board = emptyBoard(boardSize);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";
  let plies = 0;

  const agents = { white: whiteAgent, black: blackAgent };

  while (plies < maxPlies) {
    const normBoard = normalizeBoard(board, boardSize);
    const state = {
      board: normBoard,
      summonCounts: { ...summonCounts },
      currentPlayer,
      boardSize,
    };

    const turns = generateTurns(state);

    // 手がなければ相手の勝ち
    if (turns.length === 0) {
      return { winner: nextPlayer(currentPlayer), aborted: false, plies };
    }

    const agent = agents[currentPlayer];
    const chosen = agent.chooseTurn(state);

    if (!chosen) {
      return { winner: nextPlayer(currentPlayer), aborted: false, plies };
    }

    // checkWinCondition で終局確認（追加安全ネット）
    const nextState = {
      board: normalizeBoard(chosen.board, boardSize),
      summonCounts: chosen.summonCounts,
      currentPlayer: nextPlayer(currentPlayer),
      boardSize,
    };
    const meta = { maxSummons, boardSize };
    if (checkWinCondition(nextState, meta)) {
      return { winner: currentPlayer, aborted: false, plies: plies + 1 };
    }

    board = normalizeBoard(chosen.board, boardSize);
    summonCounts = chosen.summonCounts;
    currentPlayer = nextPlayer(currentPlayer);
    plies++;
  }

  return { winner: null, aborted: true, plies };
}

// ───────────────────────── マッチ実行 ─────────────────────────

/**
 * weightsA vs weightsB のマッチを実行する。
 * `games` ゲームをカラーをAB交互に入れ替えながら対戦する。
 *
 * @param {object} weightsA
 * @param {object} weightsB
 * @param {object} opts
 * @returns {{ aWins: number, bWins: number, aborted: number,
 *             decided: number, aWinRate: number|null,
 *             ci95Low: number|null, ci95High: number|null }}
 */
function runMatch(weightsA, weightsB, opts) {
  const { games, depth, timeMs, size, seed, maxPlies } = opts;

  let aWins = 0;
  let bWins = 0;
  let abortedCount = 0;

  const masterRng = mulberry32(seed);

  for (let g = 0; g < games; g++) {
    const gameSeed = (masterRng() * 0xffffffff) >>> 0;
    const gameRng = mulberry32(gameSeed);

    // カラー交互割り当て（先手有利を打ち消す）
    const aIsWhite = g % 2 === 0;

    const aRng = mulberry32((gameRng() * 0xffffffff) >>> 0);
    const bRng = mulberry32((gameRng() * 0xffffffff) >>> 0);

    const aAgent = searchPlayer(aRng, { maxDepth: depth, timeBudgetMs: timeMs, weights: weightsA });
    const bAgent = searchPlayer(bRng, { maxDepth: depth, timeBudgetMs: timeMs, weights: weightsB });

    const whiteAgent = aIsWhite ? aAgent : bAgent;
    const blackAgent = aIsWhite ? bAgent : aAgent;

    const result = playGame(size, maxPlies, whiteAgent, blackAgent);

    if (result.aborted) {
      abortedCount++;
    } else if (result.winner !== null) {
      // A が勝ったか判定
      const aWon = aIsWhite
        ? result.winner === "white"
        : result.winner === "black";
      if (aWon) aWins++;
      else bWins++;
    }
  }

  const decided = aWins + bWins;
  const aWinRate = decided > 0 ? aWins / decided : null;

  // 95% 信頼区間 (Wilson score は簡易版: ±1.96*sqrt(p(1-p)/n))
  let ci95Low = null;
  let ci95High = null;
  if (aWinRate !== null && decided > 0) {
    const margin = 1.96 * Math.sqrt((aWinRate * (1 - aWinRate)) / decided);
    ci95Low  = Math.max(0, aWinRate - margin);
    ci95High = Math.min(1, aWinRate + margin);
  }

  return { aWins, bWins, aborted: abortedCount, decided, aWinRate, ci95Low, ci95High };
}

// ───────────────────────── 出力フォーマット ─────────────────────────

function fmt(x) {
  if (x === null) return "N/A";
  return (x * 100).toFixed(1) + "%";
}

function printMatchResult(nameA, nameB, result, wallMs) {
  const { aWins, bWins, aborted, decided, aWinRate, ci95Low, ci95High } = result;
  console.log(`\n【${nameA} vs ${nameB}】`);
  console.log(`  Aウィン=${aWins}  Bウィン=${bWins}  中断=${aborted}  決着=${decided}`);
  console.log(`  A勝率: ${fmt(aWinRate)}  (95%CI: ${fmt(ci95Low)} 〜 ${fmt(ci95High)})`);

  if (aWinRate !== null) {
    if (ci95Low !== null && ci95Low > 0.50) {
      console.log(`  → ${nameA} は明らかにベース超え (CI下限 > 50%)`);
    } else if (ci95High !== null && ci95High < 0.50) {
      console.log(`  → ${nameA} は明らかにベース以下 (CI上限 < 50%)`);
    } else {
      console.log(`  → ノイズ帯内 (CIが50%を含む)`);
    }
  }
  console.log(`  経過時間: ${(wallMs / 1000).toFixed(1)} 秒`);
}

function printWeights(name, w) {
  return `${name}: top=${w.top} buried=${w.buried} material=${w.material} elimChance=${w.elimChance} mobility=${w.mobility}`;
}

// ───────────────────────── メイン ─────────────────────────

function main() {
  const args = parseArgs(process.argv.slice(2));
  const { depth, timeMs, games, size, seed, maxPlies } = args;

  console.log("=== Ukeja Layer ウェイトチューニング ===");
  console.log(`depth=${depth}  timeMs=${timeMs}  games=${games}  size=${size}  seed=${seed}  maxPlies=${maxPlies}`);
  console.log(`注意: depth=2 がデフォルト。depth=3 は遅いが精度が上がる。`);
  console.log("");

  // 候補ウェイトを表示
  console.log("=== 候補ウェイト ===");
  for (const [name, w] of Object.entries(CANDIDATES)) {
    console.log(`  ${printWeights(name, w)}`);
  }
  console.log("");

  const matchOpts = { games, depth, timeMs, size, seed, maxPlies };

  // ── 単一マッチモード ──
  if (args.a !== null && args.b !== null) {
    const wA = CANDIDATES[args.a];
    const wB = CANDIDATES[args.b];
    if (!wA) { console.error(`Unknown candidate: ${args.a}`); process.exit(1); }
    if (!wB) { console.error(`Unknown candidate: ${args.b}`); process.exit(1); }

    console.log(`単一マッチ: ${args.a} vs ${args.b} (${games}ゲーム)`);
    const t0 = Date.now();
    const result = runMatch(wA, wB, matchOpts);
    const dt = Date.now() - t0;
    printMatchResult(args.a, args.b, result, dt);
    return;
  }

  // ── デフォルト: 全候補 vs baseline のラウンド ──
  console.log(`=== ラウンド: 全候補 vs baseline (各${games}ゲーム) ===`);

  const baselineWeights = CANDIDATES.baseline;
  const results = [];

  for (const [name, weights] of Object.entries(CANDIDATES)) {
    if (name === "baseline") continue; // baseline vs baseline はスキップ

    process.stdout.write(`  ${name} vs baseline ... `);
    const t0 = Date.now();
    const result = runMatch(weights, baselineWeights, matchOpts);
    const dt = Date.now() - t0;

    results.push({ name, result, wallMs: dt });
    process.stdout.write(`完了 (${(dt / 1000).toFixed(1)}s)\n`);
  }

  // 詳細レポート
  console.log("\n=== 詳細レポート ===");
  for (const { name, result, wallMs } of results) {
    printMatchResult(name, "baseline", result, wallMs);
  }

  // サマリーテーブル（A勝率 降順）
  console.log("\n=== サマリーテーブル (A勝率 降順) ===");
  console.log(
    "候補名".padEnd(20) +
    "A勝".padStart(6) +
    "B勝".padStart(6) +
    "中断".padStart(6) +
    "A勝率".padStart(8) +
    "CI95下限".padStart(10) +
    "CI95上限".padStart(10) +
    "  評価"
  );
  console.log("─".repeat(80));

  const sorted = [...results].sort((a, b) => {
    const rA = a.result.aWinRate ?? 0;
    const rB = b.result.aWinRate ?? 0;
    return rB - rA;
  });

  for (const { name, result } of sorted) {
    const { aWins, bWins, aborted, decided, aWinRate, ci95Low, ci95High } = result;
    let eval_ = "−";
    if (aWinRate !== null) {
      if (ci95Low !== null && ci95Low > 0.50) eval_ = "★ 有望";
      else if (ci95High !== null && ci95High < 0.50) eval_ = "✗ 下回る";
      else eval_ = "〜 誤差範囲";
    }

    console.log(
      name.padEnd(20) +
      String(aWins).padStart(6) +
      String(bWins).padStart(6) +
      String(aborted).padStart(6) +
      fmt(aWinRate).padStart(8) +
      fmt(ci95Low).padStart(10) +
      fmt(ci95High).padStart(10) +
      `  ${eval_}`
    );
  }

  console.log("─".repeat(80));
  console.log("(A=候補, B=baseline)");

  // 最有望候補を抽出
  const best = sorted[0];
  if (best && best.result.ci95Low !== null && best.result.ci95Low > 0.50) {
    console.log(`\n最有望候補: ${best.name} (CI下限 ${fmt(best.result.ci95Low)} > 50%)`);
    console.log(`  ${printWeights(best.name, CANDIDATES[best.name])}`);
    console.log("  → 確認マッチを推奨: node tools/tuneWeights.mjs --a " + best.name + " --b baseline --games 150");
  } else {
    console.log("\n有意差あり候補なし → 追加調査またはウェイト維持を推奨");
  }
}

main();
