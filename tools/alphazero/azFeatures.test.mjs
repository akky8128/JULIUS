/**
 * azFeatures.test.mjs — P1: D4アクション対称変換とターン文脈ベクトルのテスト。
 *
 * 最重要: 「盤面を transformState で変換した局面での合法マイクロアクション集合 ==
 * 元局面の合法集合を transformActionIndex で写した集合」の整合テスト。
 */

import test from "node:test";
import assert from "node:assert/strict";

import { transformState, NUM_SYMMETRIES } from "../../js/ai/nnue/features.js";
import { maxSummonsFor } from "../../js/gameLogic.js";
import {
  initTurnState,
  legalMicroActions,
  applyMicroAction,
  END_TURN_ACTION,
} from "./microActions.mjs";
import {
  TURN_CONTEXT_DIM,
  TC_HEAD_OFFSET,
  TC_PHASE_OFFSET,
  TC_DIR_OFFSET,
  TC_STEPS_OFFSET,
  extractTurnContext,
  transformTurnContext,
  transformActionIndex,
  transformDir,
  NUM_MICRO_ACTIONS,
} from "./azFeatures.mjs";

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

function randomState(rng) {
  const maxSummons = maxSummonsFor(BOARD_SIZE);
  const full = rng() < 0.5;
  const summonCounts = full
    ? { white: maxSummons, black: maxSummons }
    : { white: randInt(rng, maxSummons), black: randInt(rng, maxSummons) };
  const board = Array.from({ length: BOARD_SIZE }, () =>
    Array.from({ length: BOARD_SIZE }, () => [])
  );
  for (const color of ["white", "black"]) {
    const n = randInt(rng, summonCounts[color]);
    for (let i = 0; i < n; i++) {
      board[randInt(rng, 3)][randInt(rng, 3)].push(color);
    }
  }
  // スタック内シャッフル
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

test("transformActionIndex: 全symで81ノードの全単射、sym=0は恒等", () => {
  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    const seen = new Set();
    for (let a = 0; a < NUM_MICRO_ACTIONS; a++) {
      const t = transformActionIndex(a, sym, BOARD_SIZE);
      assert.ok(t >= 0 && t < NUM_MICRO_ACTIONS, `sym=${sym} a=${a} -> ${t} 範囲外`);
      seen.add(t);
      if (sym === 0) assert.equal(t, a, `sym=0 は恒等であるべき (a=${a})`);
    }
    assert.equal(seen.size, NUM_MICRO_ACTIONS, `sym=${sym} は全単射であるべき`);
  }
});

test("transformActionIndex: 種別を保存する（移動→移動、召喚→召喚、終了→終了）", () => {
  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    for (let a = 0; a < 64; a++) assert.ok(transformActionIndex(a, sym) < 64);
    for (let a = 64; a < 80; a++) {
      const t = transformActionIndex(a, sym);
      assert.ok(t >= 64 && t < 80);
    }
    assert.equal(transformActionIndex(END_TURN_ACTION, sym), END_TURN_ACTION);
  }
});

test("transformDir: 全symで方向の全単射、sym=0は恒等", () => {
  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    const seen = new Set([0, 1, 2, 3].map((d) => transformDir(d, sym)));
    assert.equal(seen.size, 4);
    if (sym === 0) for (let d = 0; d < 4; d++) assert.equal(transformDir(d, sym), d);
  }
});

test("整合性(最重要): 変換局面の合法手集合 == 元局面の合法手集合の変換像（TURN_START＋ターン途中）", () => {
  const rng = mulberry32(42);
  const N = 200;
  for (let i = 0; i < N; i++) {
    const state = randomState(rng);
    for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
      let ctx = initTurnState(state);
      let ctxT = initTurnState(transformState(state, sym));

      // ターン途中まで追随しながら各ステップで合法手集合を比較（最大4ステップ）
      for (let step = 0; step < 4; step++) {
        const legal = legalMicroActions(ctx);
        const legalT = new Set(legalMicroActions(ctxT));
        const mapped = new Set(legal.map((a) => transformActionIndex(a, sym, BOARD_SIZE)));
        assert.deepEqual(
          [...mapped].sort((a, b) => a - b),
          [...legalT].sort((a, b) => a - b),
          `局面#${i} sym=${sym} step=${step} で合法手集合が不一致\nstate=${JSON.stringify(state)}`
        );

        // 移動系アクションを1つ選び両方に適用して追随（終了/召喚は打ち切り）
        const moveActions = legal.filter((a) => a < 64);
        if (moveActions.length === 0) break;
        const pick = moveActions[randInt(rng, moveActions.length - 1)];
        ctx = applyMicroAction(ctx, pick);
        ctxT = applyMicroAction(ctxT, transformActionIndex(pick, sym, BOARD_SIZE));
        if (ctx.done || ctxT.done) break;

        // ターン文脈ベクトルの共変性も同時に検証
        const tc = extractTurnContext(ctx);
        const tcT = extractTurnContext(ctxT);
        assert.deepEqual(
          Array.from(transformTurnContext(tc, sym, BOARD_SIZE)),
          Array.from(tcT),
          `局面#${i} sym=${sym} step=${step} でターン文脈の共変性が不一致`
        );
      }
    }
  }
});

test("extractTurnContext: TURN_STARTは head=なし・dir=なし・steps=0", () => {
  const board = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
  board[1][2] = ["white"];
  const ctx = initTurnState({
    board,
    summonCounts: { white: 1, black: 0 },
    currentPlayer: "white",
    boardSize: 4,
  });
  const vec = extractTurnContext(ctx);
  assert.equal(vec.length, TURN_CONTEXT_DIM);
  assert.equal(vec[TC_HEAD_OFFSET + 16], 1); // head なし
  assert.equal(vec[TC_PHASE_OFFSET + 0], 1); // TURN_START
  assert.equal(vec[TC_DIR_OFFSET + 4], 1); // dir なし
  assert.equal(vec[TC_STEPS_OFFSET], 0);
  assert.equal(vec.reduce((a, b) => a + b, 0), 3); // one-hot 3グループのみ
});

test("extractTurnContext: 移動後は head/フェーズ/方向/手数が反映される", () => {
  const board = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
  board[1][2] = ["white", "white"];
  let ctx = initTurnState({
    board,
    summonCounts: { white: 2, black: 0 },
    currentPlayer: "white",
    boardSize: 4,
  });
  ctx = applyMicroAction(ctx, (1 * 4 + 2) * 4 + 3); // cell(1,2), dir=3(右)
  const vec = extractTurnContext(ctx);
  assert.equal(vec[TC_HEAD_OFFSET + 1 * 4 + 2], 1); // head = A = (1,2)
  assert.equal(vec[TC_PHASE_OFFSET + 1], 1); // AUGMENT
  assert.equal(vec[TC_DIR_OFFSET + 3], 1); // dir=右
  assert.equal(vec[TC_STEPS_OFFSET], 1 / 16);
});

test("transformTurnContext: sym=0は恒等", () => {
  const board = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
  board[0][0] = ["white", "white"];
  let ctx = initTurnState({
    board,
    summonCounts: { white: 2, black: 0 },
    currentPlayer: "white",
    boardSize: 4,
  });
  ctx = applyMicroAction(ctx, 0 * 4 + 3);
  const vec = extractTurnContext(ctx);
  assert.deepEqual(Array.from(transformTurnContext(vec, 0)), Array.from(vec));
});
