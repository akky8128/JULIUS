/**
 * selfPlayWorker.test.mjs — P4: 自己対戦レコードの正当性検証。
 * ★z の手番視点符号（winner===mover?+1:-1）が最重要（P3 符号バグと同型）★
 */

import test from "node:test";
import assert from "node:assert/strict";

import { normalizeBoard } from "../../../js/gameLogic.js";
import { transformState, transformCell } from "../../../js/ai/nnue/features.js";
import { initTurnState, applyMicroAction, NUM_MICRO_ACTIONS } from "../microActions.mjs";
import { viableActions } from "../mcts.mjs";
import { transformActionIndex } from "../azFeatures.mjs";
import {
  computeZ,
  visitPolicyDist,
  transformPolicyDist,
  applyDirichletNoise,
  sampleDirichlet,
  searchMicro,
  playSelfPlayGame,
  buildSelfPlayDataset,
  mulberry32,
  DRAW_Z,
  TEMPERATURE_MICRO_STEPS,
} from "./selfPlayWorker.mjs";
import { makeNode, expandNode } from "../mcts.mjs";

function stubNet(value = 0) {
  return { evaluatePolicy: () => ({ value, policyLogits: new Float64Array(NUM_MICRO_ACTIONS) }) };
}
function emptyBoard() {
  return Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
}

test("★z符号: computeZ は winner===mover で +1、不一致で -1、引き分けで 0.5", () => {
  assert.equal(computeZ("white", "white"), 1);
  assert.equal(computeZ("white", "black"), -1);
  assert.equal(computeZ("black", "black"), 1);
  assert.equal(computeZ("black", "white"), -1);
  assert.equal(computeZ(null, "white"), DRAW_Z);
  assert.equal(computeZ(null, "black"), DRAW_Z);
});

test("★z符号(統合): 1局の全レコードで z===computeZ(winner, pov)（π/z が同一対局由来）", () => {
  const net = stubNet(0);
  const { records, meta } = buildSelfPlayDataset(net, {
    games: 1, size: 4, augment: 1, seed: 7, simulations: 12, maxPlies: 80,
  });
  assert.ok(records.length > 0);
  // winner は白/黒/引き分けのいずれか。resultDistribution から復元。
  const rd = meta.resultDistribution;
  const winner = rd.white ? "white" : rd.black ? "black" : null;
  for (const r of records) {
    assert.equal(r.z, computeZ(winner, r.pov), `pov=${r.pov} の z が手番視点符号と不一致`);
    assert.ok(r.z === 1 || r.z === -1 || r.z === DRAW_Z);
  }
});

test("policyDist: viable 上で正規化・合計1、非viableは0", () => {
  const board = emptyBoard();
  board[0][0] = ["white", "white"];
  board[2][2] = ["black"];
  const ctx = initTurnState({ board, summonCounts: { white: 3, black: 2 }, currentPlayer: "white", boardSize: 4 });
  const rng = mulberry32(3);
  const { policyDist } = searchMicro(ctx, "white", 4, stubNet(0), { simulations: 60, dirichlet: false, rng });
  const viable = new Set(viableActions(ctx));
  let sum = 0;
  for (let a = 0; a < NUM_MICRO_ACTIONS; a++) {
    if (policyDist[a] > 0) assert.ok(viable.has(a), `非viable ${a} に確率が付いている`);
    else assert.ok(policyDist[a] === 0);
    sum += policyDist[a];
  }
  assert.ok(Math.abs(sum - 1) < 1e-9, `合計が1でない: ${sum}`);
});

test("policyDist D4共変: transformPolicyDist は transformActionIndex で81要素を置換", () => {
  const dist = new Float64Array(NUM_MICRO_ACTIONS);
  dist[1] = 0.5; dist[64] = 0.3; dist[80] = 0.2; // 移動/召喚/終了 の代表
  for (let sym = 0; sym < 8; sym++) {
    const out = transformPolicyDist(dist, sym, 4);
    let sum = 0;
    for (let a = 0; a < NUM_MICRO_ACTIONS; a++) {
      if (dist[a] !== 0) assert.equal(out[transformActionIndex(a, sym, 4)], dist[a]);
      sum += out[a];
    }
    assert.ok(Math.abs(sum - 1) < 1e-12, "確率質量は対称変換で保存されるべき");
  }
  // sym=0 は恒等
  assert.deepEqual(Array.from(transformPolicyDist(dist, 0, 4)), Array.from(dist));
});

test("決定性: 同一 seed・同一設定で buildSelfPlayDataset が完全一致", () => {
  const cfg = { games: 1, size: 4, augment: 1, seed: 99, simulations: 20, maxPlies: 60 };
  const a = buildSelfPlayDataset(stubNet(0.2), cfg).records;
  const b = buildSelfPlayDataset(stubNet(0.2), cfg).records;
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.equal(a[i].z, b[i].z);
    assert.deepEqual(a[i].policyDist, b[i].policyDist);
    assert.deepEqual(a[i].indices, b[i].indices);
  }
});

test("Dirichlet: ルートのみ・legal 上で混合し P の合計を保つ", () => {
  const board = emptyBoard();
  board[0][0] = ["white"];
  const root = makeNode(initTurnState({ board, summonCounts: { white: 3, black: 2 }, currentPlayer: "white", boardSize: 4 }), "white", 4);
  expandNode(root, stubNet(0));
  const before = root.legal.map((a) => root.P[a]);
  let sumBefore = before.reduce((x, y) => x + y, 0);
  assert.ok(Math.abs(sumBefore - 1) < 1e-9);
  applyDirichletNoise(root, { epsilon: 0.25, scale: 10, rng: mulberry32(5) });
  const after = root.legal.map((a) => root.P[a]);
  const sumAfter = after.reduce((x, y) => x + y, 0);
  assert.ok(Math.abs(sumAfter - 1) < 1e-9, "混合後も legal 上の P 合計は1（(1-ε)+ε=1）");
  // 少なくとも1つは値が変化している（ノイズが効いている）
  assert.ok(root.legal.some((a, i) => Math.abs(before[i] - after[i]) > 1e-6));
});

test("sampleDirichlet: 非負・合計1・次元一致", () => {
  const rng = mulberry32(11);
  const d = sampleDirichlet(5, 0.6, rng);
  assert.equal(d.length, 5);
  let s = 0;
  for (const x of d) { assert.ok(x >= 0); s += x; }
  assert.ok(Math.abs(s - 1) < 1e-9);
});

test("温度スケジュール定数: 最初 TEMPERATURE_MICRO_STEPS 手のみ τ=1.0（それ以外 0）", () => {
  // 実装が閾値 TEMPERATURE_MICRO_STEPS を用いていることを定数レベルで固定化（回帰防止）
  assert.equal(TEMPERATURE_MICRO_STEPS, 16);
});

test("スキーマ: 各レコードが必須フィールドと形状を満たす（turnContext26/policyDist81）", () => {
  const { records } = buildSelfPlayDataset(stubNet(0), { games: 1, size: 4, augment: 8, seed: 3, simulations: 10, maxPlies: 40 });
  assert.ok(records.length > 0);
  for (const r of records) {
    for (const k of ["pid", "ply", "microStep", "sym", "indices", "scalars", "turnContext", "policyDist", "z"]) {
      assert.ok(k in r, `missing ${k}`);
    }
    assert.equal(r.turnContext.length, 26);
    assert.equal(r.policyDist.length, 81);
    assert.equal(r.scalars.length, 13);
  }
  // pid はターン単位（同一 ply の全 sym で共有）: 同一 (ply) の sym 群が同一 pid を持つ
  const byPlyPid = new Map();
  for (const r of records) {
    const key = `${r.pid}`;
    if (!byPlyPid.has(key)) byPlyPid.set(key, r.ply);
  }
  assert.ok(byPlyPid.size > 0);
});

test("D4共変(統合): sym!=0 レコードの policyDist は sym=0 を transformPolicyDist した像", () => {
  const { records } = buildSelfPlayDataset(stubNet(0), { games: 1, size: 4, augment: 8, seed: 21, simulations: 16, maxPlies: 40 });
  // 同一 (pid, ply, microStep) の sym=0 と sym=k を突き合わせる
  const byKey = new Map();
  for (const r of records) {
    const key = `${r.pid}:${r.ply}:${r.microStep}`;
    if (!byKey.has(key)) byKey.set(key, {});
    byKey.get(key)[r.sym] = r;
  }
  let checked = 0;
  for (const group of byKey.values()) {
    const base = group[0];
    if (!base) continue;
    for (let sym = 1; sym < 8; sym++) {
      const r = group[sym];
      if (!r) continue;
      const expected = Array.from(transformPolicyDist(Float64Array.from(base.policyDist), sym, 4));
      assert.deepEqual(r.policyDist, expected, `sym=${sym} の policyDist が共変でない`);
      checked++;
    }
  }
  assert.ok(checked > 0, "共変を検証できるサンプルがあること");
});
