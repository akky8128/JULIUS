/**
 * moveGen.test.mjs — generateTurns の node:test ベーステスト
 *
 * 実行: node --test tools/
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { generateTurns, hashPosition, hasAnyLegalTurn } from "../js/ai/moveGen.js";
import {
  simulateTurn,
  normalizeBoard,
  maxSummonsFor,
  checkWinCondition,
} from "../js/gameLogic.js";

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

/** generateTurns の各候補を simulateTurn でクロス検証する */
function crossValidate(state) {
  const turns = generateTurns(state);
  const errors = [];

  for (const candidate of turns) {
    let simResult;
    try {
      simResult = simulateTurn(state, candidate.actions);
    } catch (e) {
      errors.push(`actions=${JSON.stringify(candidate.actions)} → simulateTurn threw: ${e.message}`);
      continue;
    }

    // normalizeBoard でそろえて比較
    const simBoard = normalizeBoard(simResult.board, state.boardSize);
    const candBoard = candidate.board;

    const simHash = hashPosition(simBoard, simResult.summonCounts);
    const candHash = hashPosition(candBoard, candidate.summonCounts);

    if (simHash !== candHash) {
      errors.push(
        `actions=${JSON.stringify(candidate.actions)}\n` +
        `  simulateTurn hash: ${simHash}\n` +
        `  candidate hash:    ${candHash}`
      );
    }
  }

  return { turns, errors };
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

// ───────────────────────── テスト ─────────────────────────

// ── 1. 初期盤面クロス検証 ──────────────────────────────────
test("初期盤面: サモン候補を simulateTurn でクロス検証", () => {
  const state = baseState();
  const { turns, errors } = crossValidate(state);
  assert.ok(turns.length > 0, "合法手が存在するはず");
  assert.deepEqual(errors, [], `クロス検証エラー:\n${errors.join("\n")}`);
});

// ── 2. サモン中盤クロス検証 ───────────────────────────────
test("サモン中盤: 一部埋まった状態でクロス検証", () => {
  const board = emptyBoard();
  board[0][0] = ["white"];
  board[1][1] = ["black"];
  board[2][2] = ["white"];
  const state = baseState({
    board,
    summonCounts: { white: 2, black: 1 },
    currentPlayer: "black",
  });
  const { turns, errors } = crossValidate(state);
  assert.ok(turns.length > 0);
  assert.deepEqual(errors, [], errors.join("\n"));
});

// ── 3. サモン後フェーズ（混合スタック）クロス検証 ─────────
test("サモン後フェーズ: 混合スタックでクロス検証", () => {
  const board = emptyBoard();
  // 複雑なスタックを手動構築
  board[0][0] = ["black", "white"];
  board[0][1] = ["white", "black", "white"];
  board[0][2] = ["black"];
  board[1][0] = ["white", "white"];
  board[1][1] = ["black", "black", "white"];
  board[2][3] = ["white", "black"];
  board[3][3] = ["black", "white", "white"];
  const state = baseState({
    board,
    summonCounts: { white: 8, black: 8 },
    currentPlayer: "white",
  });
  const { turns, errors } = crossValidate(state);
  // エリミネートと移動が両方存在する局面
  assert.ok(turns.length > 0);
  assert.deepEqual(errors, [], errors.join("\n"));
});

// ── 4. 終盤付近クロス検証 ─────────────────────────────────
test("終盤付近: 自分のコマが少ない局面でクロス検証", () => {
  const board = emptyBoard();
  // 白が上に1枚のみ、黒が多数
  board[0][0] = ["black", "black", "white"];
  board[0][1] = ["black", "black"];
  board[1][0] = ["black", "black", "black"];
  board[1][1] = ["white"];
  const state = baseState({
    board,
    summonCounts: { white: 8, black: 8 },
    currentPlayer: "white",
  });
  const { turns, errors } = crossValidate(state);
  assert.deepEqual(errors, [], errors.join("\n"));
});

// ── 5. サモン中はエリミネートを生成しない ─────────────────
test("サモン中はエリミネートを生成しない", () => {
  const board = emptyBoard();
  board[0][0] = ["black", "white"]; // white がトップ、下に相手コマあり
  const state = baseState({
    board,
    summonCounts: { white: 1, black: 1 }, // まだサモン中
    currentPlayer: "white",
  });
  const turns = generateTurns(state);
  const hasElim = turns.some(t => t.actions.some(a => a.type === "eliminate"));
  assert.equal(hasElim, false, "サモン中にエリミネート候補があってはならない");
});

// ── 6. サモン完了後はエリミネートを生成する ───────────────
test("両者サモン完了後はエリミネートを生成する", () => {
  const board = emptyBoard();
  board[0][0] = ["black", "white"]; // white がトップ、下に相手
  const state = baseState({
    board,
    summonCounts: { white: 8, black: 8 }, // 両者完了
    currentPlayer: "white",
  });
  const turns = generateTurns(state);
  const hasElim = turns.some(t => t.actions.some(a => a.type === "eliminate"));
  assert.equal(hasElim, true, "サモン完了後はエリミネートが生成されるべき");
});

// ── 7. 連続移動制約: 前の移動先から継続し、1枚以上残す ────
test("連続移動: 移動先から続け、1枚以上残す制約", () => {
  // [0][0] に white × 3 → [0][1] に 2枚移動後、[0][1] から [0][2] へ
  const board = emptyBoard();
  board[0][0] = ["white", "white", "white"];
  const state = baseState({ board });
  const turns = generateTurns(state);

  // 2手チェーン（0,0→0,1→0,2 等）が存在するか
  const hasChain = turns.some(t =>
    t.actions.length === 2 &&
    t.actions.every(a => a.type === "move")
  );
  assert.equal(hasChain, true, "2手の連続移動チェーンが存在するべき");

  // すべての連続移動手は前の to が次の from である
  for (const t of turns) {
    const moves = t.actions.filter(a => a.type === "move");
    for (let i = 1; i < moves.length; i++) {
      const prev = moves[i - 1];
      const curr = moves[i];
      assert.equal(curr.from.r, prev.to.r, "連続移動: from.r が前の to.r と一致するべき");
      assert.equal(curr.from.c, prev.to.c, "連続移動: from.c が前の to.c と一致するべき");
    }
  }

  // クロス検証
  const { errors } = crossValidate(state);
  assert.deepEqual(errors, [], errors.join("\n"));
});

// ── 8. 重複除去: 同じ局面になる異なるチェーンは1候補 ──────
test("重複除去: 同じ結果局面は1候補に絞られる", () => {
  // 白コマが離れた2か所にあり、どちらを動かしても同じ局面になるケースを確認
  // 簡単な例: 白1枚が [0][0] にあり、右(0,1)または下(1,0)→違う局面なので
  // 同じ局面になるケースはチェーンで作る
  // [0][0]=[w,w,w], [0][1]=[w,w,w] → white のターン
  // [0][0]から2枚を[0][1]へ移動と、[0][1]から2枚を[0][0]へ移動は別局面なので
  // ここでは単純に重複除去が機能することを確認: 同じハッシュの候補が2件ない
  const board = emptyBoard();
  board[0][0] = ["white", "white", "white"];
  const state = baseState({ board });
  const turns = generateTurns(state);
  const hashes = turns.map(t => hashPosition(t.board, t.summonCounts));
  const uniqueHashes = new Set(hashes);
  assert.equal(hashes.length, uniqueHashes.size, "重複したハッシュが存在してはならない");
});

// ── 9. effective pass が候補に含まれない ──────────────────
test("effective pass (開始局面と同じ結果) が候補に含まれない", () => {
  const board = emptyBoard();
  board[0][0] = ["white"];
  const state = baseState({ board });
  const startHash = hashPosition(state.board, state.summonCounts);
  const turns = generateTurns(state);
  const hasPass = turns.some(t => hashPosition(t.board, t.summonCounts) === startHash);
  assert.equal(hasPass, false, "開始局面と同一の結果局面が候補に含まれてはならない");
});

// ── 10. hasAnyLegalTurn ────────────────────────────────────
test("hasAnyLegalTurn: サモン可能な局面で true", () => {
  const state = baseState();
  assert.equal(hasAnyLegalTurn(state), true);
});

test("hasAnyLegalTurn: 手がない局面で false", () => {
  // 白のコマがどこにもトップにない & サモン済み & 空きなし
  const board = emptyBoard();
  // 全セルに黒がトップ
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      board[r][c] = ["black"];
    }
  }
  const state = baseState({
    board,
    summonCounts: { white: 8, black: 8 },
    currentPlayer: "white",
  });
  assert.equal(hasAnyLegalTurn(state), false);
});

// ── 11. ランダム対戦クロス検証（5ゲーム × 最初30手） ───────
test("ランダム自己対戦: 各手の generateTurns 候補を simulateTurn でクロス検証", () => {
  const rng = mulberry32(12345);
  const MAX_VERIFY_PLIES = 30;

  for (let g = 0; g < 5; g++) {
    let board = emptyBoard();
    let summonCounts = { white: 0, black: 0 };
    let currentPlayer = "white";
    const maxSummons = maxSummonsFor(BOARD_SIZE);

    for (let ply = 0; ply < MAX_VERIFY_PLIES; ply++) {
      const normBoard = normalizeBoard(board, BOARD_SIZE);
      const state = { board: normBoard, summonCounts: { ...summonCounts }, currentPlayer, boardSize: BOARD_SIZE };

      const { turns, errors } = crossValidate(state);
      assert.deepEqual(
        errors, [],
        `game=${g + 1} ply=${ply}: クロス検証エラー:\n${errors.join("\n")}`
      );

      if (turns.length === 0) break;

      const chosen = turns[Math.floor(rng() * turns.length)];
      board = normalizeBoard(chosen.board, BOARD_SIZE);
      summonCounts = chosen.summonCounts;
      currentPlayer = nextPlayer(currentPlayer);
    }
  }
});
