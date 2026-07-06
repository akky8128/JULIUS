/**
 * features.test.mjs — js/ai/nnue/features.js の node:test ベーステスト
 *
 * 実行: node --test tools/nnue/features.test.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  MAX_DEPTH,
  SCALAR_DIM,
  NUM_SYMMETRIES,
  featureDim,
  extractFeatures,
  transformCell,
  transformState,
} from "../../js/ai/nnue/features.js";
import { maxSummonsFor } from "../../js/gameLogic.js";

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

// mulberry32 PRNG（evaluate.test.mjs / selfplay.mjs と同じ実装）
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 多重集合として2つの配列が一致するか調べる（ソートして比較） */
function multisetEqual(a, b) {
  if (a.length !== b.length) return false;
  const sa = [...a].sort((x, y) => x - y);
  const sb = [...b].sort((x, y) => x - y);
  return sa.every((v, i) => v === sb[i]);
}

const oppOf = (p) => (p === "white" ? "black" : "white");

// ───────────────────────── 1. index式の具体検証 ─────────────────────────

test("index式: 特定セル・特定深さ・特定色の idx が期待通り", () => {
  const board = emptyBoard();
  // (r=1, c=2) に POV=white コマを1枚だけ置く（深さ0, color=0）
  board[1][2] = ["white"];

  const state = baseState({ board, summonCounts: { white: 1, black: 0 } });
  const { indices } = extractFeatures(state, "white");

  const expectedIdx = (1 * BOARD_SIZE + 2) * MAX_DEPTH * 2 + 0 * 2 + 0;
  assert.equal(indices.length, 1, "コマは1枚だけなので indices の長さは1");
  assert.equal(indices[0], expectedIdx, "idx が式通りであること");
});

test("index式: 相手コマ(color=1)・深さ2枚目(depth=1) の idx が期待通り", () => {
  const board = emptyBoard();
  // (r=0, c=0) に black を最下段、white を2段目に積む。POV=white とすると
  // 最下段(black, i=0) は color=1・depth=0、2段目(white, i=1) は color=0・depth=1
  board[0][0] = ["black", "white"];

  const state = baseState({ board, summonCounts: { white: 1, black: 1 } });
  const { indices } = extractFeatures(state, "white");

  const idxBottom = (0 * BOARD_SIZE + 0) * MAX_DEPTH * 2 + 0 * 2 + 1; // depth0, color=1(相手)
  const idxTop = (0 * BOARD_SIZE + 0) * MAX_DEPTH * 2 + 1 * 2 + 0;    // depth1, color=0(自分)

  assert.equal(indices.length, 2, "コマは2枚なので indices の長さは2");
  assert.ok(indices.includes(idxBottom), `最下段の idx ${idxBottom} が含まれるべき`);
  assert.ok(indices.includes(idxTop), `2段目の idx ${idxTop} が含まれるべき`);
});

test("index式: 走査順は r→c→スタック下から上", () => {
  const board = emptyBoard();
  board[0][1] = ["white"];
  board[1][0] = ["black", "white"];

  const state = baseState({ board, summonCounts: { white: 2, black: 1 } });
  const { indices } = extractFeatures(state, "white");

  // (0,1) が先、(1,0) の下段→上段の順で続く
  const idx01 = (0 * BOARD_SIZE + 1) * MAX_DEPTH * 2 + 0 * 2 + 0;
  const idx10Bottom = (1 * BOARD_SIZE + 0) * MAX_DEPTH * 2 + 0 * 2 + 1;
  const idx10Top = (1 * BOARD_SIZE + 0) * MAX_DEPTH * 2 + 1 * 2 + 0;

  assert.deepEqual(indices, [idx01, idx10Bottom, idx10Top]);
});

// ───────────────────────── 2. POV不変性 ─────────────────────────

test("POV不変性: 全コマの色を反転し povPlayer も反転すると同じ特徴になる", () => {
  const rng = mulberry32(123);
  const boardA = emptyBoard();
  const piecesSpec = [
    { r: 0, c: 0, stack: ["white", "black", "white"] },
    { r: 1, c: 2, stack: ["black"] },
    { r: 3, c: 3, stack: ["white", "white"] },
    { r: 2, c: 1, stack: ["black", "black", "black", "black", "black", "black", "black", "black", "black"] }, // 9枚 (クリップ確認も兼ねる)
  ];
  for (const { r, c, stack } of piecesSpec) {
    boardA[r][c] = stack.slice();
  }

  const stateA = baseState({
    board: boardA,
    summonCounts: { white: 5, black: 6 },
    currentPlayer: "white",
  });

  // B: 色をすべて反転し、currentPlayer も反転
  const boardB = boardA.map((row) => row.map((stack) => stack.map((p) => oppOf(p))));
  const stateB = baseState({
    board: boardB,
    summonCounts: { white: stateA.summonCounts.black, black: stateA.summonCounts.white },
    currentPlayer: "black",
  });

  const featA = extractFeatures(stateA, stateA.currentPlayer);
  const featB = extractFeatures(stateB, stateB.currentPlayer);

  assert.ok(
    multisetEqual(featA.indices, featB.indices),
    `indices(多重集合)が一致しない: A=${JSON.stringify([...featA.indices].sort())}, B=${JSON.stringify([...featB.indices].sort())}`
  );
  assert.deepEqual(featA.scalars, featB.scalars, "scalars が一致しない");

  // rng を消費しておく（将来のランダム拡張に備えたプレースホルダ、決定論を崩さないことの確認）
  void rng();
});

// ───────────────────────── 3. スカラー検証 ─────────────────────────

test("スカラー: 残サモン数・サモンフェーズ終了フラグ・空セル率が式通り", () => {
  const board = emptyBoard();
  board[0][0] = ["white"];
  board[0][1] = ["black"];
  // 16セル中2セルが埋まっている

  const summonCounts = { white: 3, black: 5 };
  const state = baseState({ board, summonCounts });
  const maxS = maxSummonsFor(BOARD_SIZE);

  const { scalars } = extractFeatures(state, "white");

  assert.equal(scalars.length, SCALAR_DIM);
  assert.equal(scalars[0], (maxS - summonCounts.white) / maxS, "scalars[0]: POV側残サモン数");
  assert.equal(scalars[1], (maxS - summonCounts.black) / maxS, "scalars[1]: 相手側残サモン数");
  assert.equal(scalars[2], 0, "サモンフェーズはまだ終了していない");
  assert.equal(scalars[3], (BOARD_SIZE * BOARD_SIZE - 2) / (BOARD_SIZE * BOARD_SIZE), "scalars[3]: 空セル率");
});

test("スカラー: サモンフェーズ終了フラグが1になる", () => {
  const maxS = maxSummonsFor(BOARD_SIZE);
  const board = emptyBoard();
  const state = baseState({
    board,
    summonCounts: { white: maxS, black: maxS },
  });

  const { scalars } = extractFeatures(state, "white");
  assert.equal(scalars[2], 1, "両者サモン完了時はフラグ=1");
  assert.equal(scalars[0], 0, "残サモン数0 → scalars[0]=0");
  assert.equal(scalars[1], 0, "残サモン数0 → scalars[1]=0");
  assert.equal(scalars[3], 1, "全セル空 → 空セル率=1");
});

test("スカラー: POV を入れ替えると scalars[0]/[1] が入れ替わる", () => {
  const state = baseState({
    summonCounts: { white: 2, black: 6 },
  });

  const featWhite = extractFeatures(state, "white");
  const featBlack = extractFeatures(state, "black");

  assert.equal(featWhite.scalars[0], featBlack.scalars[1]);
  assert.equal(featWhite.scalars[1], featBlack.scalars[0]);
});

// ───────────────────────── 4. クリップ検証 ─────────────────────────

test("クリップ: MAX_DEPTHを超える枚数を積むと最上段側がMAX_DEPTH-1に畳み込まれる", () => {
  const board = emptyBoard();
  const tallStack = [];
  for (let i = 0; i < 10; i++) {
    tallStack.push(i % 2 === 0 ? "white" : "black");
  }
  board[2][3] = tallStack;

  const state = baseState({ board, summonCounts: { white: 5, black: 5 } });
  const { indices } = extractFeatures(state, "white");

  assert.equal(indices.length, 10, "積んだ枚数分だけ indices があるはず");

  const dim = featureDim(BOARD_SIZE);
  for (const idx of indices) {
    assert.ok(idx >= 0 && idx < dim, `idx ${idx} は [0, ${dim}) の範囲内であるべき`);
  }

  // 深さ MAX_DEPTH-1 以上の複数枚が同一 idx に畳み込まれていることを確認
  const cellBase = (2 * BOARD_SIZE + 3) * MAX_DEPTH * 2;
  const clippedWhiteIdx = cellBase + (MAX_DEPTH - 1) * 2 + 0;
  const clippedBlackIdx = cellBase + (MAX_DEPTH - 1) * 2 + 1;
  const countClippedWhite = indices.filter((i) => i === clippedWhiteIdx).length;
  const countClippedBlack = indices.filter((i) => i === clippedBlackIdx).length;
  // i=7,8,9 (0-indexed) が depth>=MAX_DEPTH-1=7 に畳み込まれる → i=7(white),8(black),9(white)
  assert.ok(countClippedWhite + countClippedBlack >= 2, "最上段付近が同一idxに畳み込まれ複数回出現するはず");
});

test("クリップ: 全boardSizeにわたって全idxが範囲内", () => {
  for (const boardSize of [3, 4, 5, 8]) {
    const board = Array.from({ length: boardSize }, () => Array.from({ length: boardSize }, () => []));
    const tallStack = Array.from({ length: 12 }, (_, i) => (i % 2 === 0 ? "white" : "black"));
    board[0][0] = tallStack;
    board[boardSize - 1][boardSize - 1] = ["black", "white"];

    const state = { board, summonCounts: { white: 1, black: 1 }, currentPlayer: "white", boardSize };
    const { indices } = extractFeatures(state, "white");
    const dim = featureDim(boardSize);
    for (const idx of indices) {
      assert.ok(idx >= 0 && idx < dim, `boardSize=${boardSize}: idx ${idx} out of range [0,${dim})`);
    }
  }
});

// ───────────────────────── 5. D4検証 ─────────────────────────

test("D4: 8つのsymは相異なる盤変換であり、恒等はsym=0のみ", () => {
  const boardSize = 4;
  const seenMappings = new Set();
  let identityCount = 0;

  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    const mapping = [];
    let isIdentity = true;
    const targets = new Set();
    for (let r = 0; r < boardSize; r++) {
      for (let c = 0; c < boardSize; c++) {
        const { r: nr, c: nc } = transformCell(r, c, sym, boardSize);
        assert.ok(nr >= 0 && nr < boardSize && nc >= 0 && nc < boardSize, "変換後座標は盤内であるべき");
        mapping.push(`${r},${c}->${nr},${nc}`);
        targets.add(`${nr},${nc}`);
        if (nr !== r || nc !== c) isIdentity = false;
      }
    }
    assert.equal(targets.size, boardSize * boardSize, `sym=${sym} は全単射であるべき`);
    const key = mapping.join("|");
    assert.ok(!seenMappings.has(key), `sym=${sym} は他のsymと重複している`);
    seenMappings.add(key);
    if (isIdentity) identityCount++;
  }

  assert.equal(identityCount, 1, "恒等変換はちょうど1つ(sym=0)であるべき");
  assert.equal(seenMappings.size, NUM_SYMMETRIES, "8通りすべてが相異なるべき");
});

test("D4: transformState後のextractFeaturesが座標変換したindicesの多重集合と一致する", () => {
  const boardSize = 4;
  const board = emptyBoard(boardSize);
  board[0][0] = ["white", "black", "white"];
  board[1][2] = ["black"];
  board[3][3] = ["white", "white"];
  board[2][1] = Array.from({ length: 9 }, (_, i) => (i % 2 === 0 ? "black" : "white")); // クリップも兼ねる

  const state = {
    board,
    summonCounts: { white: 4, black: 5 },
    currentPlayer: "white",
    boardSize,
  };
  const pov = "white";

  const baseFeat = extractFeatures(state, pov);
  const dim = featureDim(boardSize);

  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    const transformed = transformState(state, sym);
    const transformedFeat = extractFeatures(transformed, pov);

    // base の各 idx を「セル座標を sym 変換」した上で再構成した期待 indices を作る
    const expectedIndices = baseFeat.indices.map((idx) => {
      const cellIdx = Math.floor(idx / (MAX_DEPTH * 2));
      const rest = idx % (MAX_DEPTH * 2); // depthClipped*2 + color はそのまま
      const r = Math.floor(cellIdx / boardSize);
      const c = cellIdx % boardSize;
      const { r: nr, c: nc } = transformCell(r, c, sym, boardSize);
      return (nr * boardSize + nc) * MAX_DEPTH * 2 + rest;
    });

    assert.ok(
      multisetEqual(transformedFeat.indices, expectedIndices),
      `sym=${sym}: indices(多重集合)が一致しない`
    );
    // スカラーは盤面の座標変換に依存しないので不変のはず
    assert.deepEqual(transformedFeat.scalars, baseFeat.scalars, `sym=${sym}: scalars が変化してはならない`);

    for (const idx of transformedFeat.indices) {
      assert.ok(idx >= 0 && idx < dim, `sym=${sym}: idx ${idx} out of range`);
    }
  }
});

test("D4: transformStateは元のstateを破壊しない(イミュータブル)", () => {
  const boardSize = 4;
  const board = emptyBoard(boardSize);
  board[0][0] = ["white", "black"];
  const original = {
    board,
    summonCounts: { white: 2, black: 1 },
    currentPlayer: "white",
    boardSize,
  };
  const snapshotBoard = JSON.parse(JSON.stringify(original.board));
  const snapshotSummon = { ...original.summonCounts };

  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    const transformed = transformState(original, sym);
    // 変換結果を書き換えても元は影響を受けないことを確認
    transformed.board[0][0] = ["mutated"];
    transformed.summonCounts.white = 999;
  }

  assert.deepEqual(original.board, snapshotBoard, "元のboardが変更されてはならない");
  assert.deepEqual(original.summonCounts, snapshotSummon, "元のsummonCountsが変更されてはならない");
});

test("D4: transformStateはcurrentPlayer/boardSizeを保持する", () => {
  const boardSize = 5;
  const state = baseState({ boardSize, board: emptyBoard(boardSize), currentPlayer: "black" });
  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    const transformed = transformState(state, sym);
    assert.equal(transformed.currentPlayer, "black");
    assert.equal(transformed.boardSize, boardSize);
  }
});

// ───────────────────────── featureDim / 定数の健全性 ─────────────────────────

test("featureDim: boardSize=4 で 4*4*MAX_DEPTH*2", () => {
  assert.equal(featureDim(4), 4 * 4 * MAX_DEPTH * 2);
});

test("定数: MAX_DEPTH, SCALAR_DIM, NUM_SYMMETRIES の値", () => {
  assert.equal(MAX_DEPTH, 8);
  assert.equal(SCALAR_DIM, 4);
  assert.equal(NUM_SYMMETRIES, 8);
});
