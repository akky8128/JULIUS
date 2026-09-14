/**
 * turnToMicro.test.mjs — P2: 分解→再生一致のプロパティテスト。
 * ランダム局面 × generateTurns の全候補で decomposeTurn が
 * (1) 各ステップ合法、(2) 最終盤面一致、を満たすことを検証する。
 */

import test from "node:test";
import assert from "node:assert/strict";

import { generateTurns, MAX_CANDIDATES } from "../../js/ai/moveGen.js";
import { maxSummonsFor } from "../../js/gameLogic.js";
import { decomposeTurn, turnActionsToMicroIndices } from "./turnToMicro.mjs";
import { END_TURN_ACTION, SUMMON_ACTIONS_START } from "./microActions.mjs";

const BOARD_SIZE = 4;

function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randInt(rng, maxInclusive) {
  return Math.floor(rng() * (maxInclusive + 1));
}

function randomState(rng, phaseHint) {
  const maxSummons = maxSummonsFor(BOARD_SIZE);
  let summonCounts;
  let cap;
  if (phaseHint === "early") {
    summonCounts = { white: randInt(rng, 4), black: randInt(rng, 4) };
    cap = { white: summonCounts.white, black: summonCounts.black };
  } else if (phaseHint === "mid") {
    summonCounts = { white: maxSummons, black: maxSummons };
    cap = { white: maxSummons, black: maxSummons };
  } else {
    summonCounts = { white: maxSummons, black: maxSummons };
    const low = Math.max(1, Math.floor(maxSummons / 3));
    cap = { white: low, black: low };
  }
  const board = Array.from({ length: BOARD_SIZE }, () =>
    Array.from({ length: BOARD_SIZE }, () => [])
  );
  for (const color of ["white", "black"]) {
    const n = randInt(rng, cap[color]);
    for (let i = 0; i < n; i++) board[randInt(rng, 3)][randInt(rng, 3)].push(color);
  }
  for (const row of board) {
    for (const stack of row) {
      for (let i = stack.length - 1; i > 0; i--) {
        const j = randInt(rng, i);
        [stack[i], stack[j]] = [stack[j], stack[i]];
      }
    }
  }
  return {
    board,
    summonCounts,
    currentPlayer: rng() < 0.5 ? "white" : "black",
    boardSize: BOARD_SIZE,
  };
}

test("turnActionsToMicroIndices: 召喚/排除は単独ノード、移動は count 回複製 + END", () => {
  assert.deepEqual(
    turnActionsToMicroIndices([{ type: "summon", r: 1, c: 2 }], BOARD_SIZE),
    [SUMMON_ACTIONS_START + 6]
  );
  assert.deepEqual(
    turnActionsToMicroIndices(
      [{ type: "move", from: { r: 0, c: 0 }, to: { r: 0, c: 1 }, count: 3 }],
      BOARD_SIZE
    ),
    [3, 3, 3, END_TURN_ACTION] // cell0*4+dir3(右) ×3 + END
  );
  assert.deepEqual(
    turnActionsToMicroIndices(
      [
        { type: "move", from: { r: 0, c: 0 }, to: { r: 0, c: 1 }, count: 2 },
        { type: "move", from: { r: 0, c: 1 }, to: { r: 1, c: 1 }, count: 1 },
      ],
      BOARD_SIZE
    ),
    [3, 3, 1 * 4 + 1, END_TURN_ACTION] // (0,0)右×2, (0,1)=cell1 下×1, END
  );
});

test("decomposeTurn: 非隣接moveや複合summonはErrorを投げる", () => {
  assert.throws(
    () =>
      turnActionsToMicroIndices(
        [{ type: "move", from: { r: 0, c: 0 }, to: { r: 2, c: 0 }, count: 1 }],
        BOARD_SIZE
      ),
    /non-adjacent/
  );
  assert.throws(
    () =>
      turnActionsToMicroIndices(
        [
          { type: "summon", r: 0, c: 0 },
          { type: "summon", r: 0, c: 1 },
        ],
        BOARD_SIZE
      ),
    /only action/
  );
});

test("プロパティ: ランダム局面×全候補で分解→再生一致（各ステップ合法＋最終盤面一致）", () => {
  const N = 300;
  const rng = mulberry32(20260714);
  let positions = 0;
  let candidatesChecked = 0;
  let cappedSkipped = 0;
  let microStepsTotal = 0;

  for (let i = 0; i < N; i++) {
    const phaseHint = ["early", "mid", "late"][i % 3];
    const state = randomState(rng, phaseHint);
    const candidates = generateTurns(state);
    if (candidates.length >= MAX_CANDIDATES) {
      cappedSkipped++;
      continue;
    }
    positions++;
    for (const candidate of candidates) {
      // decomposeTurn は違反時に throw する（それがそのままテスト失敗になる）
      const { actionIndices, contexts } = decomposeTurn(state, candidate);
      assert.equal(actionIndices.length, contexts.length);
      assert.ok(actionIndices.length >= 1);
      candidatesChecked++;
      microStepsTotal += actionIndices.length;
    }
  }

  console.log(
    `[turnToMicro] positions=${positions}/${N} candidates=${candidatesChecked} ` +
      `avgMicroSteps=${(microStepsTotal / Math.max(1, candidatesChecked)).toFixed(2)} cappedSkipped=${cappedSkipped}`
  );
  assert.ok(candidatesChecked > 1000, "十分な候補数を検証できるべき");
});
