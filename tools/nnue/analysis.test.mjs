// analysis.test.mjs — Phase 1: analyzeTopMoves (js/ai/nnue/analysis.js) のテスト
//
// 実行方法: node --test tools/nnue/analysis.test.mjs
//
// models/gen071.json / models/gen071.bin を実際にロードして検証する
// （parity.test.mjs と同様、fileURLToPath でプロジェクトルートを解決する）。

import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadNetwork } from "../../js/ai/nnue/network.js";
import { analyzeTopMoves } from "../../js/ai/nnue/analysis.js";
import { maxSummonsFor } from "../../js/gameLogic.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..", "..");
const jsonPath = path.join(projectRoot, "models", "gen071.json");

// 探索を軽量にするための共通オプション（テスト実行時間短縮のため maxDepth を抑える）
const FAST_OPTIONS = { maxDepth: 3, timeBudgetMs: 500, topN: 3, rng: () => 0 };

/** 中盤局面: 序盤サモンを数手進めた盤面（denormalized: 空セル=0）。 */
function midgameState() {
  return {
    boardSize: 4,
    currentPlayer: "white",
    summonCounts: { white: 3, black: 3 },
    board: [
      [["white"], ["black"], ["white"], ["black"]],
      [["white"], ["black"], 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ],
  };
}

/**
 * 即勝ち局面: サモン完了済みで盤上に白1枚・黒1枚のみ。
 * 白が唯一の合法手として黒の上に乗り、黒を完全に埋める(盤上に黒の駒が
 * 他になくなる)ため checkWinCondition が true になる。
 */
function forcedWinState() {
  const maxSummons = maxSummonsFor(4);
  return {
    boardSize: 4,
    currentPlayer: "white",
    summonCounts: { white: maxSummons, black: maxSummons },
    board: [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, ["white"], ["black"]],
    ],
  };
}

test("analyzeTopMoves: gen071 をロードして実行できる", async () => {
  const net = await loadNetwork(jsonPath);
  const result = analyzeTopMoves(midgameState(), net, FAST_OPTIONS);
  assert.ok(result);
  assert.ok(Array.isArray(result.candidates));
});

test("analyzeTopMoves: 中盤局面で candidates が仕様を満たす", async () => {
  const net = await loadNetwork(jsonPath);
  const result = analyzeTopMoves(midgameState(), net, FAST_OPTIONS);

  assert.ok(result.candidates.length <= 3);
  assert.ok(result.candidates.length > 0);

  for (const cand of result.candidates) {
    assert.equal(typeof cand.notation, "string");
    assert.equal(typeof cand.score, "number");
    assert.ok(Number.isFinite(cand.score));
    assert.ok(cand.winProb >= 0 && cand.winProb <= 1);
    assert.ok(Array.isArray(cand.board));
    assert.ok(Array.isArray(cand.actions));
  }

  // score 降順にソートされていること
  for (let i = 1; i < result.candidates.length; i++) {
    assert.ok(result.candidates[i - 1].score >= result.candidates[i].score);
  }

  // current.score は最上位候補の score と一致する
  assert.equal(result.current.score, result.candidates[0].score);
  assert.ok(result.current.winProb >= 0 && result.current.winProb <= 1);
  assert.ok(result.current.whiteWinProb >= 0 && result.current.whiteWinProb <= 1);
});

test("analyzeTopMoves: 決定性（同じ rng・同じ局面なら同一の score 列）", async () => {
  const net = await loadNetwork(jsonPath);
  const state = midgameState();

  const result1 = analyzeTopMoves(state, net, FAST_OPTIONS);
  const result2 = analyzeTopMoves(state, net, FAST_OPTIONS);

  const scores1 = result1.candidates.map((c) => c.score);
  const scores2 = result2.candidates.map((c) => c.score);
  assert.deepEqual(scores1, scores2);
});

test("analyzeTopMoves: 即勝ち局面で isWin と notation の # を検証", async () => {
  const net = await loadNetwork(jsonPath);
  const result = analyzeTopMoves(forcedWinState(), net, FAST_OPTIONS);

  assert.ok(result.candidates.length > 0);
  const winCand = result.candidates[0];
  assert.equal(winCand.isWin, true);
  assert.ok(winCand.notation.endsWith("#"));
  assert.equal(result.current.winProb, 1 / (1 + Math.exp(-winCand.score)));
});

test("analyzeTopMoves: 合法手がない局面では candidates が空配列", async () => {
  const net = await loadNetwork(jsonPath);
  const maxSummons = maxSummonsFor(4);
  // 白のコマが盤上に無く、かつサモンも使い切っている(合法手なし)状態を構成。
  const state = {
    boardSize: 4,
    currentPlayer: "white",
    summonCounts: { white: maxSummons, black: maxSummons },
    board: [
      [["black"], ["black"], ["black"], ["black"]],
      [["black"], ["black"], ["black"], ["black"]],
      [["black"], ["black"], ["black"], ["black"]],
      [["black"], ["black"], ["black"], ["black"]],
    ],
  };
  const result = analyzeTopMoves(state, net, FAST_OPTIONS);
  assert.deepEqual(result.candidates, []);
  assert.equal(result.current.score, 0);
});
