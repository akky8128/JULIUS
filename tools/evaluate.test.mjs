/**
 * evaluate.test.mjs — evaluate.js / players.js の node:test ベーステスト
 *
 * 実行: node --test tools/evaluate.test.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { evaluate, WEIGHTS, WIN_SCORE } from "../js/ai/evaluate.js";
import { randomPlayer, greedyPlayer } from "../js/ai/players.js";
import { checkWinCondition, maxSummonsFor, normalizeBoard } from "../js/gameLogic.js";

// ───────────────────────── ヘルパー ─────────────────────────

const BOARD_SIZE = 4;

function emptyBoard(size = BOARD_SIZE) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => []));
}

function baseState(overrides = {}) {
  return {
    board: emptyBoard(),
    summonCounts: { white: 0, black: 0 },
    currentPlayer: "white",
    boardSize: BOARD_SIZE,
    ...overrides,
  };
}

// mulberry32 PRNG（selfplay.mjs と同じ実装）
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function nextPlayer(p) {
  return p === "white" ? "black" : "white";
}

// ───────────────────────── evaluate() の項目テスト ─────────────────────────

// ── 1. top 項目: white が 3セル, black が 1セルのトップを持つ ──
test("top 項目: white=3top black=1top → white 視点で正", () => {
  const board = emptyBoard();
  board[0][0] = ["white"];
  board[0][1] = ["white"];
  board[1][0] = ["white"];
  board[1][1] = ["black"];

  const state = baseState({
    board,
    summonCounts: { white: 3, black: 1 },
  });

  // top の差分は white 3 - black 1 = +2
  const scoreForWhite = evaluate(state, "white");
  assert.ok(scoreForWhite > 0, `white 視点スコアは正のはず: ${scoreForWhite}`);

  // top の差分は black 1 - white 3 = -2
  const scoreForBlack = evaluate(state, "black");
  assert.ok(scoreForBlack < 0, `black 視点スコアは負のはず: ${scoreForBlack}`);

  // top の差分だけで見ると white = +2, black = -2
  // WEIGHTS.top=10 なら top 項の寄与は ±20
  const topDiff = (3 - 1) * WEIGHTS.top;
  // material も変わるので完全一致はしないが、top 項の符号は必ず反映される
  assert.ok(scoreForWhite - scoreForBlack > 0, "white スコアは black スコアより大きいはず");
});

// ── 2. buried 項目: ["black","white"] → white が black を埋めている ──
test('buried 項目: stack=["black","white"] → white 視点で buried +1', () => {
  const board = emptyBoard();
  board[0][0] = ["black", "white"]; // white がトップ、black が埋まっている

  const state = baseState({
    board,
    summonCounts: { white: 8, black: 8 },
  });

  const scoreWhite = evaluate(state, "white");
  const scoreBlack = evaluate(state, "black");

  // white は: top+1, buried+1, elimChance+1
  // black は: top=0, buried=0, elimChance=0
  // white 視点スコアは正のはず
  assert.ok(scoreWhite > 0, `white 視点スコアは正のはず: ${scoreWhite}`);
  assert.ok(scoreBlack < 0, `black 視点スコアは負のはず: ${scoreBlack}`);
});

// ── 3. elimChance 項目: ["black","white"] → white の elimChance = 1 ──
test('elimChance 項目: stack=["black","white"] → white に elimChance', () => {
  const board = emptyBoard();
  board[0][0] = ["black", "white"];

  // サモン完了後（elimChance が実際に機能する）
  const state = baseState({
    board,
    summonCounts: { white: 8, black: 8 },
  });

  const scoreWhite = evaluate(state, "white");
  // サモン完了 → elimChance も top も white が有利 → 正
  assert.ok(scoreWhite > 0, `white 視点スコアは正のはず: ${scoreWhite}`);
});

// ── 4. material 項目: 一方がコマを多く持つ ──
test("material 項目: 残サモン差が反映される", () => {
  const board = emptyBoard();
  // white が 1枚すでにサモン、black は 0枚
  board[0][0] = ["white"];

  const stateMore = baseState({
    board,
    summonCounts: { white: 1, black: 0 },
  });

  // white: 盤上1 + 残サモン(8-1)=7 = 8
  // black: 盤上0 + 残サモン(8-0)=8 = 8
  // material 差は 0 (残サモン数で相殺)
  // ただし top 差は white +1 → white 有利
  const scoreWhite = evaluate(stateMore, "white");
  assert.ok(scoreWhite > 0, `白がトップ1つ多い→ white 視点スコア正: ${scoreWhite}`);
});

// ── 5. 対称性: evaluate(state, 'white') === -evaluate(state, 'black') ──
test("対称性: evaluate(state,'white') === -evaluate(state,'black')", () => {
  const board = emptyBoard();
  board[0][0] = ["white"];
  board[0][1] = ["black", "white"];
  board[1][0] = ["white", "black"];
  board[2][2] = ["black"];

  const state = baseState({
    board,
    summonCounts: { white: 4, black: 3 },
  });

  const sw = evaluate(state, "white");
  const sb = evaluate(state, "black");
  assert.equal(sw, -sb, `対称性が成立していない: sw=${sw}, sb=${sb}`);
});

// ── 6. 対称性2: 複数のランダム局面で対称性を確認 ──
test("対称性: ランダム局面 10 個で evaluate(state,'white') === -evaluate(state,'black')", () => {
  const rng = mulberry32(999);

  for (let i = 0; i < 10; i++) {
    const board = emptyBoard();
    const pieces = ["white", "black", "white", "black", "white"];
    for (const p of pieces) {
      const r = Math.floor(rng() * BOARD_SIZE);
      const c = Math.floor(rng() * BOARD_SIZE);
      board[r][c].push(p);
    }

    const state = baseState({
      board,
      summonCounts: { white: 3, black: 2 },
    });

    const sw = evaluate(state, "white");
    const sb = evaluate(state, "black");
    assert.equal(sw, -sb, `局面 ${i}: 対称性が成立していない sw=${sw}, sb=${sb}`);
  }
});

// ── 7. mobility 項目: スタックに自コマが多いほどモビリティ高い ──
test("mobility 項目: 自コマが多いスタックほど高スコア", () => {
  const board1 = emptyBoard();
  board1[0][0] = ["white"]; // ownTop=1

  const board3 = emptyBoard();
  board3[0][0] = ["white", "white", "white"]; // ownTop=3

  const state1 = baseState({ board: board1, summonCounts: { white: 8, black: 8 } });
  const state3 = baseState({ board: board3, summonCounts: { white: 8, black: 8 } });

  const s1 = evaluate(state1, "white");
  const s3 = evaluate(state3, "white");

  // ownTop が大きいほど mobility が高い → スコアが高いはず
  assert.ok(s3 > s1, `ownTop=3 の方がスコア高いはず: s3=${s3}, s1=${s1}`);
});

// ── 8. WIN_SCORE がエクスポートされている ──
test("WIN_SCORE がエクスポートされている", () => {
  assert.equal(typeof WIN_SCORE, "number");
  assert.ok(WIN_SCORE >= 1e6, `WIN_SCORE は十分大きいはず: ${WIN_SCORE}`);
});

// ───────────────────────── greedyPlayer のテスト ─────────────────────────

// ── 9. 即勝ち優先: 勝ち手がある局面では必ずそれを選ぶ ──
test("greedyPlayer: 即勝ち候補がある場合はそれを選ぶ", () => {
  // white がトップに立つコマが全セルにある（つまり black は次の手番で手がない）状態を作る
  // ただし generateTurns で候補が出るよう、move で勝てる状況を使う
  //
  // 簡単な設定:
  // サモン完了後、black のコマはすべて white の下に埋まっている → checkWinCondition=true
  // white に1手動かす余地があれば、その手でも「既に勝ち」なので
  // 厳密に「その1手が即勝ち」となる局面を構築する
  //
  // 具体的: white が1コマをあるセルに移動すると black が全滅 → 相手に手なし
  // ここでは: セルに["black","white","white"]→ white が top2枚移動後、
  // もとのセルに["black"]のみ残る → black が top → 手あり、なので難しい
  //
  // 代わりに: white が eliminate して black のコマをすべて除去する局面
  //   board[0][0] = ["black","white"] (white top, black below)
  //   残り全セル空 → white eliminate → board[0][0]=["white"] → black に合法手なし
  const board = emptyBoard();
  board[0][0] = ["black", "white"];
  // summonCounts both=8 (サモン完了)
  const state = baseState({
    board,
    summonCounts: { white: 8, black: 8 },
    currentPlayer: "white",
  });

  const rng = mulberry32(42);
  const greedy = greedyPlayer(rng);

  const chosen = greedy.chooseTurn(state);
  assert.ok(chosen !== null, "chooseTurn は null を返してはならない");

  // 選んだ手の結果局面で black に手がないか確認（checkWinCondition）
  const maxSummons = maxSummonsFor(BOARD_SIZE);
  const meta = { maxSummons, boardSize: BOARD_SIZE };
  const nextState = {
    board: chosen.board,
    summonCounts: chosen.summonCounts,
    currentPlayer: "black",
    boardSize: BOARD_SIZE,
  };
  const isWin = checkWinCondition(nextState, meta);
  assert.equal(isWin, true, "greedyPlayer は即勝ち候補を選ぶべき");
});

// ── 10. greedy vs random 勝率スモークテスト ──
test("greedy vs random: 30ゲーム中 greedy が 80% 以上の勝率 (seed=42)", { timeout: 120000 }, () => {
  const masterRng = mulberry32(42);
  const maxSummons = maxSummonsFor(BOARD_SIZE);
  const maxPlies = 400;
  const numGames = 30;

  let greedyWins = 0;
  let decided = 0;

  for (let g = 0; g < numGames; g++) {
    const gameSeed = (masterRng() * 0xffffffff) >>> 0;
    const gameRng = mulberry32(gameSeed);

    // 前半: white=greedy, black=random
    // 後半: white=random, black=greedy
    const isSwapped = g >= Math.floor(numGames / 2);

    const greedyRng = mulberry32((gameRng() * 0xffffffff) >>> 0);
    const randomRng = mulberry32((gameRng() * 0xffffffff) >>> 0);

    const whiteAgent = isSwapped ? randomPlayer(randomRng) : greedyPlayer(greedyRng);
    const blackAgent = isSwapped ? greedyPlayer(greedyRng) : randomPlayer(randomRng);

    let board = Array.from({ length: BOARD_SIZE }, () => Array.from({ length: BOARD_SIZE }, () => []));
    let summonCounts = { white: 0, black: 0 };
    let currentPlayer = "white";
    let plies = 0;
    let winner = null;

    while (plies < maxPlies) {
      const normBoard = normalizeBoard(board, BOARD_SIZE);
      const state = { board: normBoard, summonCounts: { ...summonCounts }, currentPlayer, boardSize: BOARD_SIZE };

      const agent = currentPlayer === "white" ? whiteAgent : blackAgent;
      const chosen = agent.chooseTurn(state);

      if (!chosen) {
        winner = currentPlayer === "white" ? "black" : "white";
        break;
      }

      board = normalizeBoard(chosen.board, BOARD_SIZE);
      summonCounts = chosen.summonCounts;
      currentPlayer = currentPlayer === "white" ? "black" : "white";
      plies++;
    }

    if (winner !== null) {
      decided++;
      // greedy が勝ったか判定
      const isGreedyColor = isSwapped
        ? winner === "black"   // swap 後半は black=greedy
        : winner === "white";  // swap 前半は white=greedy
      if (isGreedyColor) greedyWins++;
    }
  }

  const winRate = decided > 0 ? greedyWins / decided : 0;
  console.log(`    greedy 勝利: ${greedyWins}/${decided} (勝率 ${(winRate * 100).toFixed(1)}%)`);

  assert.ok(
    decided > 0,
    "少なくとも1ゲーム決着がついていないとテストできない"
  );
  assert.ok(
    winRate >= 0.80,
    `greedy の勝率が 80% 以上であるべき: ${(winRate * 100).toFixed(1)}% (${greedyWins}/${decided})`
  );
});
