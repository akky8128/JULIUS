// parity.test.mjs — Step 4: JS 推論エンジンと Python 学習パイプラインの
// golden データとの一致(parity)を検証する。
//
// 実行方法: node --test tools/nnue/parity.test.mjs
//
// mock は使わず、実際に Step 3 で生成された models/gen001.json /
// models/gen001.bin / models/gen001.golden.json を読み込んで検証する。

import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadNetwork, createNetwork } from "../../js/ai/nnue/network.js";
import { extractFeatures } from "../../js/ai/nnue/features.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..", "..");
const jsonPath = path.join(projectRoot, "models", "gen001.json");
const goldenPath = path.join(projectRoot, "models", "gen001.golden.json");

const TOLERANCE = 1e-4;

test("loadNetwork: models/gen001.json をロードできる(Nodeパス)", async () => {
  const network = await loadNetwork(jsonPath);
  assert.equal(typeof network.evaluate, "function");
  assert.equal(typeof network.evaluateSigmoid, "function");
  assert.equal(typeof network.evaluateState, "function");
  assert.equal(network.meta.format, "nnue-ukeja-v1");
});

test("parity: golden 全サンプルで evaluate/evaluateSigmoid が一致する", async () => {
  const network = await loadNetwork(jsonPath);
  const golden = JSON.parse(await readFile(goldenPath, "utf8"));

  assert.ok(Array.isArray(golden.samples) && golden.samples.length > 0);

  let maxOutDiff = 0;
  let maxSigmoidDiff = 0;

  for (const sample of golden.samples) {
    const out = network.evaluate(sample.indices, sample.scalars);
    const sigmoid = network.evaluateSigmoid(sample.indices, sample.scalars);

    const outDiff = Math.abs(out - sample.out);
    const sigmoidDiff = Math.abs(sigmoid - sample.sigmoid);

    maxOutDiff = Math.max(maxOutDiff, outDiff);
    maxSigmoidDiff = Math.max(maxSigmoidDiff, sigmoidDiff);

    assert.ok(
      outDiff < TOLERANCE,
      `out mismatch: expected ${sample.out}, got ${out} (diff=${outDiff})`
    );
    assert.ok(
      sigmoidDiff < TOLERANCE,
      `sigmoid mismatch: expected ${sample.sigmoid}, got ${sigmoid} (diff=${sigmoidDiff})`
    );
  }

  console.log(
    `[parity] samples=${golden.samples.length} maxOutDiff=${maxOutDiff.toExponential(3)} maxSigmoidDiff=${maxSigmoidDiff.toExponential(3)}`
  );
});

test("createNetwork: 重み長さが meta と一致しない場合は Error を投げる", async () => {
  const meta = JSON.parse(await readFile(jsonPath, "utf8"));
  const wrongLength = new Float32Array(10); // 明らかに短すぎる
  assert.throws(() => createNetwork(wrongLength, meta), Error);
});

test("evaluateState: extractFeatures 経由の evaluate と一致する", async () => {
  const network = await loadNetwork(jsonPath);

  // 簡単な手構築 state（3x3スタックのミニ局面）
  const state = {
    boardSize: 4,
    currentPlayer: "white",
    summonCounts: { white: 2, black: 3 },
    board: [
      [["white"], [], [], []],
      [[], ["black", "white"], [], []],
      [[], [], [], []],
      [[], [], [], ["white", "white", "black"]],
    ],
  };

  const { indices, scalars } = extractFeatures(state, "white");
  const expected = network.evaluate(indices, scalars);
  const actual = network.evaluateState(state, "white");

  assert.equal(actual, expected);

  // povPlayer 省略時は state.currentPlayer を使う
  const actualDefaultPov = network.evaluateState(state);
  assert.equal(actualDefaultPov, expected);
});

test("empty indices のサンプルも正しく評価できる", async () => {
  const network = await loadNetwork(jsonPath);
  const golden = JSON.parse(await readFile(goldenPath, "utf8"));

  const emptySample = golden.samples.find(
    (s) => Array.isArray(s.indices) && s.indices.length === 0
  );
  assert.ok(emptySample, "golden データに empty indices のサンプルが存在すること");

  const out = network.evaluate(emptySample.indices, emptySample.scalars);
  const sigmoid = network.evaluateSigmoid(
    emptySample.indices,
    emptySample.scalars
  );

  assert.ok(Math.abs(out - emptySample.out) < TOLERANCE);
  assert.ok(Math.abs(sigmoid - emptySample.sigmoid) < TOLERANCE);
});
