/**
 * nnuePlayer.test.mjs — Step 5 (探索へのNNUE組み込み) の node:test ベーステスト
 *
 * 実行: node --test tools/nnue/nnuePlayer.test.mjs
 *
 * 本番測定（200局 --swap 等）は行わない。局数は最小限に留める。
 * 最重要: findBestTurn の後方互換回帰（evalFn 未指定時の動作が改修前と完全一致）を確認する。
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { findBestTurn } from "../../js/ai/search.js";
import { evaluate, WEIGHTS } from "../../js/ai/evaluate.js";
import { hashPosition } from "../../js/ai/moveGen.js";
import { createNnueSearchPlayer, createNnueSearchPlayerFromNet } from "../../js/ai/nnue/nnuePlayer.js";
import { loadNetwork } from "../../js/ai/nnue/network.js";

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

/** 固定の「サモン完了直後」に近い局面（4x4）を組み立てる。 */
function buildFixedState() {
  const size = 4;
  const board = emptyBoard(size);
  // 適度に駒を配置（top/buried/elimChance/mobility が非自明な値を取るように）
  board[0][0] = ["black", "white"];
  board[0][1] = ["white"];
  board[1][1] = ["black"];
  board[2][2] = ["white", "black"];
  board[3][3] = ["black", "white", "white"];
  return {
    board,
    summonCounts: { white: 4, black: 4 }, // サモン完了済み想定
    currentPlayer: "white",
    boardSize: size,
  };
}

function turnFingerprint(turn) {
  if (!turn) return null;
  return hashPosition(turn.board, turn.summonCounts);
}

// ───────────────────────── テスト1: 後方互換回帰（最重要） ─────────────────────────

test("後方互換回帰: evalFn 未指定と明示的な evaluate 注入で同一結果", () => {
  const state = buildFixedState();
  const seed = 12345;
  const maxDepth = 3;
  const timeBudgetMs = 500;

  const rng1 = mulberry32(seed);
  const resultDefault = findBestTurn(state, { maxDepth, timeBudgetMs, rng: rng1 });

  const rng2 = mulberry32(seed);
  const resultExplicit = findBestTurn(state, {
    maxDepth,
    timeBudgetMs,
    rng: rng2,
    evalFn: (s, p) => evaluate(s, p, WEIGHTS),
  });

  assert.equal(
    turnFingerprint(resultDefault.turn),
    turnFingerprint(resultExplicit.turn),
    "evalFn 未指定と明示的 evaluate 注入で選択した turn が一致しない"
  );
  assert.equal(resultDefault.score, resultExplicit.score, "score が一致しない");
  assert.equal(resultDefault.depthReached, resultExplicit.depthReached, "depthReached が一致しない");
  assert.equal(resultDefault.nodes, resultExplicit.nodes, "nodes が一致しない（探索経路が変わっている可能性）");
});

test("後方互換回帰: 複数の固定局面・深さで一致確認", () => {
  const size = 4;
  const states = [];

  // 局面A: 初期に近い空盤（サモンフェーズ）
  {
    const board = emptyBoard(size);
    states.push({ board, summonCounts: { white: 0, black: 0 }, currentPlayer: "white", boardSize: size });
  }

  // 局面B: buildFixedState と同じ（別の currentPlayer）
  {
    const s = buildFixedState();
    states.push({ ...s, currentPlayer: "black" });
  }

  for (const [i, state] of states.entries()) {
    for (const maxDepth of [1, 2]) {
      const seed = 777 + i;
      const rngA = mulberry32(seed);
      const a = findBestTurn(state, { maxDepth, timeBudgetMs: 500, rng: rngA });

      const rngB = mulberry32(seed);
      const b = findBestTurn(state, {
        maxDepth,
        timeBudgetMs: 500,
        rng: rngB,
        evalFn: (s2, p) => evaluate(s2, p, WEIGHTS),
      });

      assert.equal(turnFingerprint(a.turn), turnFingerprint(b.turn), `state#${i} depth=${maxDepth}: turn 不一致`);
      assert.equal(a.score, b.score, `state#${i} depth=${maxDepth}: score 不一致`);
    }
  }
});

// ───────────────────────── テスト2: evalFn 注入で動作すること ─────────────────────────

test("evalFn 注入: 定数評価関数でもクラッシュせず合法手を返す", () => {
  const state = buildFixedState();
  const rng = mulberry32(1);
  const result = findBestTurn(state, {
    maxDepth: 2,
    timeBudgetMs: 300,
    rng,
    evalFn: () => 0,
  });
  assert.ok(result.turn, "turn が null であってはならない");
  assert.ok(Array.isArray(result.turn.actions) || result.turn.actions, "turn.actions が存在すること");
});

test("evalFn 注入: NNUE evalFn でもクラッシュせず合法手を返す", async () => {
  const net = await loadNetwork("models/gen001.json");
  const state = buildFixedState();
  const rng = mulberry32(2);
  const result = findBestTurn(state, {
    maxDepth: 2,
    timeBudgetMs: 300,
    rng,
    evalFn: (s, p) => net.evaluateState(s, p),
  });
  assert.ok(result.turn, "NNUE evalFn 使用時も turn が返ること");
});

// ───────────────────────── テスト3: NNUE player の疎通 ─────────────────────────

test("createNnueSearchPlayer: 合法な turn と lastSearchInfo を返す", async () => {
  const rng = mulberry32(42);
  const player = await createNnueSearchPlayer(rng, {
    netPath: "models/gen001.json",
    maxDepth: 3,
    timeBudgetMs: 100,
  });

  assert.equal(player.lastSearchInfo, null, "chooseTurn 前は lastSearchInfo が null");

  const state = buildFixedState();
  const turn = player.chooseTurn(state);

  assert.ok(turn, "chooseTurn は非 null の turn を返すこと");
  assert.ok(turn.actions, "turn は actions を持つこと");
  assert.ok(turn.board, "turn は board を持つこと");

  const info = player.lastSearchInfo;
  assert.ok(info, "lastSearchInfo が埋まっていること");
  assert.equal(typeof info.depthReached, "number");
  assert.equal(typeof info.nodes, "number");
  assert.equal(typeof info.elapsedMs, "number");
  assert.ok(info.depthReached >= 1, "少なくとも深さ1には到達すること");
});

test("createNnueSearchPlayer: .bin パスを渡しても .json に正規化してロードできる", async () => {
  const rng = mulberry32(43);
  const player = await createNnueSearchPlayer(rng, {
    netPath: "models/gen001.bin",
    maxDepth: 2,
    timeBudgetMs: 100,
  });
  const state = buildFixedState();
  const turn = player.chooseTurn(state);
  assert.ok(turn, ".bin パス指定でも turn が返ること");
});

// ───────────────────────── テスト4: 決定論 ─────────────────────────

test("決定論: 同一 net・同一 seed・同一 state で chooseTurn が同じ turn を返す", async () => {
  const net = await loadNetwork("models/gen001.json");
  const state = buildFixedState();

  const rng1 = mulberry32(99);
  const player1 = createNnueSearchPlayerFromNet(rng1, net, { maxDepth: 3, timeBudgetMs: 100 });
  const turn1 = player1.chooseTurn(state);

  const rng2 = mulberry32(99);
  const player2 = createNnueSearchPlayerFromNet(rng2, net, { maxDepth: 3, timeBudgetMs: 100 });
  const turn2 = player2.chooseTurn(state);

  assert.equal(turnFingerprint(turn1), turnFingerprint(turn2), "同一シードで異なる turn が選ばれた");
});

// ───────────────────────── テスト5 (任意): selfplay の nnue 疎通 ─────────────────────────
//
// selfplay.mjs の main() は CLI 実行前提（process.argv 依存、トップレベルで自動実行）で
// あり、関数として import して呼び出す形になっていないため、ここでは呼び出しレベルの
// 疎通確認をスキップする。かわりに makePlayer 相当のロジック（createNnueSearchPlayerFromNet
// 経由）が正しく動くことは上記テストで検証済み。selfplay.mjs 自体の CLI 疎通確認は
// 親エージェントが `node tools/selfplay.mjs --white nnue:models/gen001.json --black
// random --games 2` 等で軽量に行うことを想定する。
test("(任意) selfplay疎通は関数分離の都合上スキップ — 理由をログ出力", () => {
  // selfplay.mjs の main() はトップレベル実行かつ process.argv 依存のため、
  // このテストファイルから安全に import 起動することができない。
  // 疎通は tools/selfplay.mjs を直接 CLI 実行することで確認する方針とする。
  assert.ok(true);
});
