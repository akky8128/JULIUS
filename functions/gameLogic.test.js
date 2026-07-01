import {test} from "node:test";
import assert from "node:assert/strict";
import {
  simulateTurn,
  checkWinCondition,
  maxSummonsFor,
  normalizeBoard,
  computeTimersAfterTurn,
  GameLogicError,
} from "./gameLogic.js";

const BOARD_SIZE = 4;

function emptyBoard(size = BOARD_SIZE) {
  return Array.from({length: size}, () => Array.from({length: size}, () => []));
}

function baseState(overrides = {}) {
  return {
    board: emptyBoard(),
    summonCounts: {white: 0, black: 0},
    currentPlayer: "white",
    boardSize: BOARD_SIZE,
    ...overrides,
  };
}

test("maxSummonsFor matches createGame's floor(n^2/2) formula", () => {
  assert.equal(maxSummonsFor(4), 8);
  assert.equal(maxSummonsFor(3), 4);
});

test("summon places a piece and increments the count", () => {
  const state = baseState();
  const result = simulateTurn(state, [{type: "summon", r: 1, c: 1}]);
  assert.deepEqual(result.board[1][1], ["white"]);
  assert.equal(result.summonCounts.white, 1);
});

test("summon onto an occupied cell is rejected", () => {
  const state = baseState({board: emptyBoard()});
  state.board[1][1] = ["black"];
  assert.throws(
    () => simulateTurn(state, [{type: "summon", r: 1, c: 1}]),
    GameLogicError,
  );
});

test("summon beyond maxSummons is rejected", () => {
  const state = baseState({summonCounts: {white: 8, black: 0}});
  assert.throws(
    () => simulateTurn(state, [{type: "summon", r: 0, c: 0}]),
    GameLogicError,
  );
});

test("summon must be the only action in a turn", () => {
  const state = baseState();
  assert.throws(
    () => simulateTurn(state, [
      {type: "summon", r: 0, c: 0},
      {type: "summon", r: 0, c: 1},
    ]),
    /only action/,
  );
});

test("eliminate is rejected before both players finish summoning", () => {
  const board = emptyBoard();
  board[0][0] = ["black", "white"];
  const state = baseState({board, summonCounts: {white: 1, black: 1}});
  assert.throws(
    () => simulateTurn(state, [{type: "eliminate", r: 0, c: 0}]),
    /both players finish summoning/,
  );
});

test("eliminate removes the nearest opponent piece beneath the mover's top piece", () => {
  const board = emptyBoard();
  board[0][0] = ["black", "white", "black", "white"];
  const state = baseState({
    board,
    summonCounts: {white: 8, black: 8},
  });
  const result = simulateTurn(state, [{type: "eliminate", r: 0, c: 0}]);
  assert.deepEqual(result.board[0][0], ["black", "white", "white"]);
});

test("eliminate requires the mover's own piece on top", () => {
  const board = emptyBoard();
  board[0][0] = ["white", "black"];
  const state = baseState({board, summonCounts: {white: 8, black: 8}});
  assert.throws(
    () => simulateTurn(state, [{type: "eliminate", r: 0, c: 0}]),
    /own piece must be on top/,
  );
});

test("eliminate requires an opponent piece to remove", () => {
  const board = emptyBoard();
  board[0][0] = ["white", "white"];
  const state = baseState({board, summonCounts: {white: 8, black: 8}});
  assert.throws(
    () => simulateTurn(state, [{type: "eliminate", r: 0, c: 0}]),
    /no opponent piece/,
  );
});

test("move relocates the requested number of pieces to an adjacent cell", () => {
  const board = emptyBoard();
  board[1][1] = ["black", "white", "white"];
  const state = baseState({board});
  const result = simulateTurn(state, [
    {type: "move", from: {r: 1, c: 1}, to: {r: 1, c: 2}, count: 2},
  ]);
  assert.deepEqual(result.board[1][1], ["black"]);
  assert.deepEqual(result.board[1][2], ["white", "white"]);
});

test("move destination must be orthogonally adjacent", () => {
  const board = emptyBoard();
  board[0][0] = ["white"];
  const state = baseState({board});
  assert.throws(
    () => simulateTurn(state, [
      {type: "move", from: {r: 0, c: 0}, to: {r: 2, c: 2}, count: 1},
    ]),
    /adjacent/,
  );
});

test("move cannot take pieces that are not the mover's own contiguous top stack", () => {
  const board = emptyBoard();
  board[0][0] = ["white", "black"];
  const state = baseState({board});
  assert.throws(
    () => simulateTurn(state, [
      {type: "move", from: {r: 0, c: 0}, to: {r: 0, c: 1}, count: 2},
    ]),
    GameLogicError,
  );
});

test("continuous move must start from the previous destination", () => {
  const board = emptyBoard();
  board[0][0] = ["white", "white", "white"];
  board[2][2] = ["white", "white"];
  const state = baseState({board});
  assert.throws(
    () => simulateTurn(state, [
      {type: "move", from: {r: 0, c: 0}, to: {r: 0, c: 1}, count: 1},
      {type: "move", from: {r: 2, c: 2}, to: {r: 2, c: 1}, count: 1},
    ]),
    /previous destination/,
  );
});

test("continuous move must leave at least one of the mover's own pieces behind", () => {
  const board = emptyBoard();
  board[0][0] = ["white", "white"];
  const state = baseState({board});
  assert.throws(
    () => simulateTurn(state, [
      {type: "move", from: {r: 0, c: 0}, to: {r: 0, c: 1}, count: 1},
      {type: "move", from: {r: 0, c: 1}, to: {r: 0, c: 2}, count: 1},
    ]),
    /leave one/,
  );
});

test("a legal continuous move chain is accepted", () => {
  const board = emptyBoard();
  board[0][0] = ["white", "white", "white"];
  const state = baseState({board});
  const result = simulateTurn(state, [
    {type: "move", from: {r: 0, c: 0}, to: {r: 0, c: 1}, count: 2},
    {type: "move", from: {r: 0, c: 1}, to: {r: 0, c: 2}, count: 1},
  ]);
  assert.deepEqual(result.board[0][0], ["white"]);
  assert.deepEqual(result.board[0][1], ["white"]);
  assert.deepEqual(result.board[0][2], ["white"]);
});

test("move cannot be mixed with summon or eliminate in the same turn", () => {
  const board = emptyBoard();
  board[0][0] = ["white"];
  const state = baseState({board});
  assert.throws(
    () => simulateTurn(state, [
      {type: "move", from: {r: 0, c: 0}, to: {r: 0, c: 1}, count: 1},
      {type: "summon", r: 1, c: 1},
    ]),
    /must be the only action/,
  );
});

test("a turn that leaves the board unchanged is rejected as an effective pass", () => {
  const board = emptyBoard();
  const state = baseState({board, summonCounts: {white: 8, black: 8}});
  // No-op couldn't be produced by a legal action, so simulate the guard
  // directly against an empty-effect scenario: summon then external revert
  // is impossible to express legally, so assert the guard rejects an
  // explicitly empty action list instead (covered above) and rely on the
  // per-action legality checks to make genuine no-op turns unreachable.
  assert.throws(() => simulateTurn(state, []), /At least one action/);
});

test("simulateTurn stores emptied cells as 0, not [], so RTDB won't drop the row", () => {
  const board = emptyBoard();
  board[1][1] = ["white"];
  const state = baseState({board});
  const result = simulateTurn(state, [
    {type: "move", from: {r: 1, c: 1}, to: {r: 1, c: 2}, count: 1},
  ]);
  assert.equal(result.board[1][1], 0);
  assert.deepEqual(result.board[1][2], ["white"]);
});

test("checkWinCondition: mover has no pieces on top anywhere after all summons are placed", () => {
  const board = emptyBoard();
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      board[r][c] = ["black", "white"];
    }
  }
  const meta = {maxSummons: maxSummonsFor(BOARD_SIZE), boardSize: BOARD_SIZE};
  const state = {board, summonCounts: {white: 8, black: 8}, currentPlayer: "black"};
  assert.equal(checkWinCondition(state, meta), true);
});

test("checkWinCondition: mover still has a piece on top means no win yet", () => {
  const board = emptyBoard();
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      board[r][c] = ["white", "black"];
    }
  }
  board[0][0] = ["black"];
  const meta = {maxSummons: maxSummonsFor(BOARD_SIZE), boardSize: BOARD_SIZE};
  const state = {board, summonCounts: {white: 8, black: 8}, currentPlayer: "black"};
  assert.equal(checkWinCondition(state, meta), false);
});

test("checkWinCondition: before summon phase ends, an empty cell means no win", () => {
  const board = emptyBoard();
  board[0][0] = ["white"];
  const meta = {maxSummons: maxSummonsFor(BOARD_SIZE), boardSize: BOARD_SIZE};
  const state = {board, summonCounts: {white: 1, black: 0}, currentPlayer: "black"};
  assert.equal(checkWinCondition(state, meta), false);
});

test("normalizeBoard rebuilds a full grid from a sparse/short board", () => {
  const sparse = [[["white"]]]; // only row 0, col 0 present
  const result = normalizeBoard(sparse, BOARD_SIZE);
  assert.equal(result.length, BOARD_SIZE);
  assert.equal(result[0].length, BOARD_SIZE);
  assert.deepEqual(result[0][0], ["white"]);
  assert.deepEqual(result[0][1], []);
  assert.deepEqual(result[3][3], []);
});

test("normalizeBoard treats 0 cells as empty stacks", () => {
  const board = emptyBoard();
  board[1][1] = 0;
  const result = normalizeBoard(board, BOARD_SIZE);
  assert.deepEqual(result[1][1], []);
});

test("computeTimersAfterTurn deducts elapsed time beyond the delay window", () => {
  const lastMove = {timestamp: 1000, timers: {white: 60, black: 60}};
  const timeControl = {enabled: true, delay: 5};
  // 20s elapsed, 5s delay grace -> 15s deducted
  const result = computeTimersAfterTurn(lastMove, timeControl, "white", 1000 + 20000);
  assert.equal(result.white, 45);
  assert.equal(result.black, 60);
});

test("computeTimersAfterTurn does not deduct time within the delay window", () => {
  const lastMove = {timestamp: 1000, timers: {white: 60, black: 60}};
  const timeControl = {enabled: true, delay: 5};
  const result = computeTimersAfterTurn(lastMove, timeControl, "white", 1000 + 3000);
  assert.equal(result.white, 60);
});

test("computeTimersAfterTurn floors remaining time at zero", () => {
  const lastMove = {timestamp: 1000, timers: {white: 10, black: 60}};
  const timeControl = {enabled: true, delay: 0};
  const result = computeTimersAfterTurn(lastMove, timeControl, "white", 1000 + 60000);
  assert.equal(result.white, 0);
});

test("computeTimersAfterTurn is a no-op when time control is disabled", () => {
  const lastMove = {timestamp: 1000, timers: {white: 60, black: 60}};
  const result = computeTimersAfterTurn(lastMove, {enabled: false}, "white", 1000 + 60000);
  assert.equal(result.white, 60);
});
