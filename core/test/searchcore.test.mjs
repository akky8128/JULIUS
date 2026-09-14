/**
 * searchcore.test.mjs — Zobrist ハッシュ・make/unmake・探索コアの整合性テスト
 *
 * 実行: node --test core/test/searchcore.test.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  createZobrist,
  hashFromScratch,
  hashToKey,
} from "../ai/zobrist.js";
import { applyTurn } from "../ai/searchGen.js";
import { generateTurns, hashPosition } from "../ai/moveGen.js";
import { findBestTurn } from "../ai/search.js";
import { evaluate, WEIGHTS } from "../ai/evaluate.js";
import { maxSummonsFor, normalizeBoard } from "../gameLogic.js";

// ───────────────────────── ヘルパー ─────────────────────────

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

function cloneBoard(board) {
  return board.map((row) => row.map((stack) => stack.slice()));
}

function nextPlayer(p) {
  return p === "white" ? "black" : "white";
}

/**
 * ランダムに合法手を積み重ねて局面列を作る。
 * サモン優先→やがて移動/エリミネートも混じる自然な進行にする。
 * @returns {Array<{board, summonCounts, currentPlayer, boardSize}>} 各手番前の局面列
 */
function randomGameStates(boardSize, rng, numStates) {
  const states = [];
  let board = emptyBoard(boardSize);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";

  for (let i = 0; i < numStates; i++) {
    const state = {
      board: normalizeBoard(board, boardSize),
      summonCounts: { ...summonCounts },
      currentPlayer,
      boardSize,
    };
    states.push(state);

    const turns = generateTurns(state);
    if (turns.length === 0) break;
    const pick = turns[Math.floor(rng() * turns.length)];
    board = pick.board;
    summonCounts = pick.summonCounts;
    currentPlayer = nextPlayer(currentPlayer);
  }
  return states;
}

/** 連続移動チェーンが必ず発生するように仕込んだ局面を作る */
function chainHeavyState(boardSize) {
  const board = emptyBoard(boardSize);
  const maxSummons = maxSummonsFor(boardSize);
  // white の駒を積み上げて連続移動チェーンを誘発
  board[0][0] = ["white", "white", "white"];
  board[1][1] = ["white", "white"];
  board[2][2] = ["black", "black"];
  board[0][2] = ["black"];
  return {
    board,
    summonCounts: { white: maxSummons, black: maxSummons },
    currentPlayer: "white",
    boardSize,
  };
}

/** board+summonCounts の正規化キー（順序不問集合比較用） */
function resultKey(board, summonCounts) {
  return hashPosition(board, summonCounts);
}

// ───────────────────────── 1. Zobrist 整合性 ─────────────────────────

test("Zobrist: from-scratch と make/unmake の差分更新が一致する（複数手適用）", () => {
  const boardSize = 4;
  const zctx = createZobrist(boardSize);
  const rng = mulberry32(123);

  const states = randomGameStates(boardSize, rng, 12);
  assert.ok(states.length > 3, "十分な数の局面が生成されているはず");

  // 各局面で from-scratch ハッシュを計算し、直前の可変盤面上での適用結果と比較する
  let board = cloneBoard(states[0].board);
  let summonCounts = { ...states[0].summonCounts };
  let player = states[0].currentPlayer;
  let hash = hashFromScratch(board, summonCounts, player, zctx);

  for (let i = 0; i + 1 < states.length; i++) {
    const fromState = { board, summonCounts, currentPlayer: player, boardSize };
    const turns = generateTurns(fromState);
    assert.ok(turns.length > 0, "合法手があるはず");

    // states[i+1] に対応する候補を探す（同一結果になる手を1つ選ぶ）
    const targetKey = resultKey(states[i + 1].board, states[i + 1].summonCounts);
    let matched = null;
    for (const t of turns) {
      if (resultKey(t.board, t.summonCounts) === targetKey) {
        matched = t;
        break;
      }
    }
    // ランダム進行と再生の手が必ずしも一致しないケースがあるため、
    // 一致しない場合は turns[0] を代わりに適用して独立に検証する。
    const cand = matched || turns[0];

    // 適用前のスナップショット（undo 後の完全復帰確認用）
    const boardBeforeApply = cloneBoard(board);
    const scBeforeApply = { ...summonCounts };

    const { hash: newHash, undo } = applyTurn(board, summonCounts, cand.actions, player, hash, zctx);
    const scratch = hashFromScratch(board, summonCounts, nextPlayer(player), zctx);
    assert.equal(hashToKey(newHash), hashToKey(scratch),
      "差分更新後のハッシュは from-scratch 再計算と一致するはず");

    // unmake で完全復帰することを確認
    undo();
    assert.deepEqual(board, boardBeforeApply, "undo 後 board が適用前と完全一致するはず");
    assert.deepEqual(summonCounts, scBeforeApply, "undo 後 summonCounts が適用前と完全一致するはず");
    const scratchBack = hashFromScratch(board, summonCounts, player, zctx);
    assert.equal(hashToKey(scratchBack), hashToKey(hash),
      "unmake 後のハッシュは適用前のハッシュに戻るはず");

    // 実際に進める（今度は undo せず次のループへ引き継ぐ）
    const applied = applyTurn(board, summonCounts, cand.actions, player, hash, zctx);
    hash = applied.hash;
    player = nextPlayer(player);
  }
});

test("Zobrist: 異なる summonCounts / 手番で異なるハッシュになる", () => {
  const boardSize = 4;
  const zctx = createZobrist(boardSize);
  const board = emptyBoard(boardSize);
  board[0][0] = ["white"];

  const h1 = hashFromScratch(board, { white: 1, black: 0 }, "white", zctx);
  const h2 = hashFromScratch(board, { white: 2, black: 0 }, "white", zctx);
  const h3 = hashFromScratch(board, { white: 1, black: 0 }, "black", zctx);
  const h4 = hashFromScratch(board, { white: 1, black: 1 }, "white", zctx);

  assert.notEqual(hashToKey(h1), hashToKey(h2), "summonCounts.white 差で異なるハッシュ");
  assert.notEqual(hashToKey(h1), hashToKey(h3), "手番差で異なるハッシュ");
  assert.notEqual(hashToKey(h1), hashToKey(h4), "summonCounts.black 差で異なるハッシュ");
});

// ───────────────────────── 2. ムーブ生成 equivalence ─────────────────────────

test("equivalence: 到達局面集合が generateTurns と一致する（boardSize=4, ランダム局面）", () => {
  const rng = mulberry32(2024);
  const boardSize = 4;

  for (let trial = 0; trial < 15; trial++) {
    const states = randomGameStates(boardSize, rng, 8);
    const state = states[states.length - 1];

    const expected = generateTurns(state);
    const expectedKeys = new Set(expected.map((t) => resultKey(t.board, t.summonCounts)));

    // 「新ムーブ生成」= generateTurns 自体を使い、make/unmake で再生できることを検証
    // (本実装では searchGen.applyTurn が候補の actions を可変盤面に正しく適用できることを
    //  到達局面のハッシュ一致で確認する)
    const board = cloneBoard(state.board);
    const summonCounts = { ...state.summonCounts };
    const zctx = createZobrist(boardSize);
    const baseHash = hashFromScratch(board, summonCounts, state.currentPlayer, zctx);

    const actualKeys = new Set();
    for (const cand of expected) {
      const { undo } = applyTurn(board, summonCounts, cand.actions, state.currentPlayer, baseHash, zctx);
      actualKeys.add(resultKey(board, summonCounts));
      undo();
      // undo 後は元通りであること
      assert.equal(resultKey(board, summonCounts), resultKey(state.board, state.summonCounts));
    }

    assert.deepEqual(
      [...actualKeys].sort(),
      [...expectedKeys].sort(),
      `trial=${trial}: make/unmake 再生後の到達局面集合が generateTurns と一致するはず`
    );
  }
});

test("equivalence: 連続移動チェーンが発生する局面での一致確認", () => {
  const boardSize = 4;
  const state = chainHeavyState(boardSize);

  const expected = generateTurns(state);
  assert.ok(expected.length > 0);
  // チェーンが実際に発生していること（actions.length > 1 の候補が存在）
  assert.ok(expected.some((t) => t.actions.length > 1), "連続移動チェーンの候補が含まれるはず");

  const zctx = createZobrist(boardSize);
  const board = cloneBoard(state.board);
  const summonCounts = { ...state.summonCounts };
  const baseHash = hashFromScratch(board, summonCounts, state.currentPlayer, zctx);

  const expectedKeys = new Set(expected.map((t) => resultKey(t.board, t.summonCounts)));
  const actualKeys = new Set();

  for (const cand of expected) {
    const { undo } = applyTurn(board, summonCounts, cand.actions, state.currentPlayer, baseHash, zctx);
    actualKeys.add(resultKey(board, summonCounts));
    undo();
  }

  assert.deepEqual([...actualKeys].sort(), [...expectedKeys].sort());
});

test("equivalence: boardSize=5 のランダム局面でも一致する", () => {
  const rng = mulberry32(55);
  const boardSize = 5;

  for (let trial = 0; trial < 8; trial++) {
    const states = randomGameStates(boardSize, rng, 10);
    const state = states[states.length - 1];

    const expected = generateTurns(state);
    const expectedKeys = new Set(expected.map((t) => resultKey(t.board, t.summonCounts)));

    const zctx = createZobrist(boardSize);
    const board = cloneBoard(state.board);
    const summonCounts = { ...state.summonCounts };
    const baseHash = hashFromScratch(board, summonCounts, state.currentPlayer, zctx);

    const actualKeys = new Set();
    for (const cand of expected) {
      const { undo } = applyTurn(board, summonCounts, cand.actions, state.currentPlayer, baseHash, zctx);
      actualKeys.add(resultKey(board, summonCounts));
      undo();
    }

    assert.deepEqual([...actualKeys].sort(), [...expectedKeys].sort());
  }
});

// ───────────────────────── 3. make/unmake 復元 ─────────────────────────

test("make/unmake: ランダムターン適用後の undo で board/summonCounts が完全一致", () => {
  const boardSize = 4;
  const rng = mulberry32(99);
  const zctx = createZobrist(boardSize);

  const states = randomGameStates(boardSize, rng, 10);

  for (const state of states) {
    const turns = generateTurns(state);
    if (turns.length === 0) continue;

    const board = cloneBoard(state.board);
    const summonCounts = { ...state.summonCounts };
    const baseHash = hashFromScratch(board, summonCounts, state.currentPlayer, zctx);

    for (const cand of turns.slice(0, Math.min(turns.length, 10))) {
      const boardSnapshot = cloneBoard(board);
      const scSnapshot = { ...summonCounts };

      const { undo } = applyTurn(board, summonCounts, cand.actions, state.currentPlayer, baseHash, zctx);
      undo();

      assert.deepEqual(board, boardSnapshot, "undo 後 board が完全一致するはず");
      assert.deepEqual(summonCounts, scSnapshot, "undo 後 summonCounts が完全一致するはず");
    }
  }
});

// ───────────────────────── 4. findBestTurn 妥当性・回帰 ─────────────────────────

test("findBestTurn: 返す turn が generateTurns(state) の合法手集合に含まれる", () => {
  const boardSize = 4;
  const board = emptyBoard(boardSize);
  board[0][0] = ["white"];
  board[0][1] = ["black"];
  board[1][0] = ["black", "white"];
  board[1][1] = ["white", "black"];
  board[2][2] = ["white"];
  board[3][3] = ["black"];

  const state = {
    board,
    summonCounts: { white: 4, black: 4 },
    currentPlayer: "white",
    boardSize,
  };

  const rng = mulberry32(42);
  const result = findBestTurn(state, { maxDepth: 3, timeBudgetMs: 2000, rng });

  assert.ok(result.turn !== null);
  const legalKeys = new Set(generateTurns(state).map((t) => resultKey(t.board, t.summonCounts)));
  assert.ok(
    legalKeys.has(resultKey(result.turn.board, result.turn.summonCounts)),
    "findBestTurn の返す turn は合法手集合に含まれるはず"
  );
});

test("findBestTurn: forwardPruneWidth:0 でも壊れず合法手を返す", () => {
  const boardSize = 4;
  const board = emptyBoard(boardSize);
  board[0][0] = ["white"];
  board[0][1] = ["black"];
  board[0][2] = ["white", "black"];
  board[1][0] = ["black", "white"];
  board[1][1] = ["white"];
  board[2][2] = ["black"];

  const state = {
    board,
    summonCounts: { white: 5, black: 5 },
    currentPlayer: "white",
    boardSize,
  };

  const rng = mulberry32(7);
  const result = findBestTurn(state, { maxDepth: 3, timeBudgetMs: 3000, rng, forwardPruneWidth: 0 });

  assert.ok(result.turn !== null);
  const legalKeys = new Set(generateTurns(state).map((t) => resultKey(t.board, t.summonCounts)));
  assert.ok(legalKeys.has(resultKey(result.turn.board, result.turn.summonCounts)));
});

test("findBestTurn: evalFn 未指定と明示的な evaluate 注入で同一結果", () => {
  const boardSize = 4;
  const board = emptyBoard(boardSize);
  board[0][0] = ["white"];
  board[0][1] = ["black"];
  board[1][0] = ["black", "white"];
  board[1][1] = ["white", "black"];

  const state = {
    board,
    summonCounts: { white: 4, black: 4 },
    currentPlayer: "white",
    boardSize,
  };

  const opts = { maxDepth: 3, timeBudgetMs: 2000 };
  const r1 = findBestTurn(state, { ...opts, rng: mulberry32(1) });
  const r2 = findBestTurn(state, {
    ...opts,
    rng: mulberry32(1),
    evalFn: (s, p) => evaluate(s, p, WEIGHTS),
  });

  assert.equal(
    resultKey(r1.turn.board, r1.turn.summonCounts),
    resultKey(r2.turn.board, r2.turn.summonCounts),
    "evalFn 省略時と明示注入時で同じ手を返すはず"
  );
  assert.equal(r1.score, r2.score);
});
