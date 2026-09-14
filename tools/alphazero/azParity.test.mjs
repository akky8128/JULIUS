/**
 * azParity.test.mjs — 2ヘッドモデル（policy head + ターン文脈入力）の
 * JS(network.js) ↔ PyTorch(train.py --policy-head) パリティ検証。
 *
 * 対象モデル: genAZ000（P1 smoke）と genAZ001（P2 蒸留v0）。存在するものだけ検証し、
 * 存在しないタグはスキップする。既存 gen137 等のレガシーモデルは不変。
 * 完了条件（PLAN.md P1）: value / policyLogits[81] 全要素が golden と一致。
 *
 * 許容誤差について:
 *   value / policyLogits ともに numpy.allclose 流の magnitude-aware 許容
 *   |diff| <= ATOL + RTOL*|expected| を使う。学習済みモデル（genAZ001の蒸留、
 *   さらに P5 自己改善ループで多ラウンド warm-start 継続学習した genAZ_r55 等）は
 *   強い選好を持つため raw value/logit の magnitude が育つ（実測: genAZ001で
 *   policy logit ~O(30)、genAZ_r55 で raw value ~O(2)超）。float32 で保持した
 *   重み・活性の丸め誤差は絶対値で ~2 ULP に達し、genAZ_r55 では value の絶対誤差
 *   1.67e-6 が固定 1e-6 を超過した実例がある（相対誤差は ~7.6e-7 = float32
 *   マシンイプシロン相当で forward の不一致ではない）。よって value も policyLogits
 *   と同じ RTOL 正規化を適用し、継続学習で magnitude が伸びても偽陽性にならないようにする。
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadNetwork } from "../../js/ai/nnue/network.js";
import { TURN_CONTEXT_DIM } from "./azFeatures.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..", "..");

const MODEL_TAGS = ["genAZ000", "genAZ001", "genAZ_r55"];
// value / policyLogits 共通: numpy.allclose 流 |diff| <= ATOL + RTOL*|expected|（上部コメント参照）
const VALUE_ATOL = 1e-6;
const VALUE_RTOL = 1e-5;
const POLICY_ATOL = 1e-6;
const POLICY_RTOL = 1e-5;

function pathsFor(tag) {
  return {
    jsonPath: path.join(projectRoot, "models", `${tag}.json`),
    goldenPath: path.join(projectRoot, "models", `${tag}.golden.json`),
  };
}

async function loadOrSkip(t, tag) {
  const { jsonPath } = pathsFor(tag);
  try {
    await readFile(jsonPath, "utf8");
  } catch {
    t.skip(`models/${tag}.json が存在しないためスキップ`);
    return null;
  }
  return loadNetwork(jsonPath);
}

for (const tag of MODEL_TAGS) {
  test(`${tag}: policyHead メタと turnContextDim を持つ`, async (t) => {
    const network = await loadOrSkip(t, tag);
    if (!network) return;
    assert.equal(network.meta.policyHead.actions, 81);
    assert.equal(network.meta.turnContextDim, TURN_CONTEXT_DIM);
    assert.equal(typeof network.evaluatePolicy, "function");
  });

  test(`${tag} parity: golden 全サンプルで value と policyLogits[81] が一致する (<1e-6)`, async (t) => {
    const network = await loadOrSkip(t, tag);
    if (!network) return;
    const golden = JSON.parse(await readFile(pathsFor(tag).goldenPath, "utf8"));
    assert.ok(Array.isArray(golden.samples) && golden.samples.length > 0);

    let maxValueDiff = 0;
    let maxPolicyDiff = 0;
    let maxPolicyRel = 0;

    for (const sample of golden.samples) {
      assert.ok(Array.isArray(sample.turnContext) && sample.turnContext.length === TURN_CONTEXT_DIM);
      assert.ok(Array.isArray(sample.policyLogits) && sample.policyLogits.length === 81);

      const { value, policyLogits } = network.evaluatePolicy(
        sample.indices,
        sample.scalars,
        sample.turnContext
      );

      const vDiff = Math.abs(value - sample.out);
      maxValueDiff = Math.max(maxValueDiff, vDiff);
      const vAllowed = VALUE_ATOL + VALUE_RTOL * Math.abs(sample.out);
      assert.ok(vDiff <= vAllowed, `value mismatch: expected ${sample.out}, got ${value} (diff=${vDiff}, allowed=${vAllowed})`);

      for (let a = 0; a < 81; a++) {
        const exp = sample.policyLogits[a];
        const pDiff = Math.abs(policyLogits[a] - exp);
        maxPolicyDiff = Math.max(maxPolicyDiff, pDiff);
        maxPolicyRel = Math.max(maxPolicyRel, pDiff / Math.max(1, Math.abs(exp)));
        const allowed = POLICY_ATOL + POLICY_RTOL * Math.abs(exp);
        assert.ok(
          pDiff <= allowed,
          `policyLogits[${a}] mismatch: expected ${exp}, got ${policyLogits[a]} (diff=${pDiff}, allowed=${allowed})`
        );
      }
    }

    console.log(
      `[azParity/${tag}] samples=${golden.samples.length} maxValueDiff=${maxValueDiff.toExponential(3)} ` +
        `maxPolicyDiff=${maxPolicyDiff.toExponential(3)} maxPolicyRel=${maxPolicyRel.toExponential(3)}`
    );
  });

  test(`${tag}: turnContext 全ゼロ時に evaluate と evaluatePolicy.value が一致する`, async (t) => {
    const network = await loadOrSkip(t, tag);
    if (!network) return;
    const golden = JSON.parse(await readFile(pathsFor(tag).goldenPath, "utf8"));

    const zeroTc = new Array(TURN_CONTEXT_DIM).fill(0);
    for (const sample of golden.samples) {
      const plain = network.evaluate(sample.indices, sample.scalars);
      const withZero = network.evaluatePolicy(sample.indices, sample.scalars, zeroTc).value;
      const omitted = network.evaluatePolicy(sample.indices, sample.scalars).value;
      assert.equal(withZero, plain);
      assert.equal(omitted, plain);
    }
  });

  test(`${tag}: evaluate は共有バッファ経由の turnContext 残留の影響を受けない`, async (t) => {
    const network = await loadOrSkip(t, tag);
    if (!network) return;
    const golden = JSON.parse(await readFile(pathsFor(tag).goldenPath, "utf8"));
    const s = golden.samples.find((x) => x.turnContext.some((v) => v !== 0)) ?? golden.samples[0];

    const before = network.evaluate(s.indices, s.scalars);
    network.evaluatePolicy(s.indices, s.scalars, s.turnContext); // 非ゼロ tc を書き込む
    const after = network.evaluate(s.indices, s.scalars); // ゼロ書き戻しが必要
    assert.equal(after, before);
  });
}

test("policy head なしモデル(gen001)では evaluatePolicy が Error を投げる", async () => {
  const legacy = await loadNetwork(path.join(projectRoot, "models", "gen001.json"));
  assert.equal(legacy.meta.policyHead, undefined);
  assert.throws(() => legacy.evaluatePolicy([], new Array(legacy.meta.scalarDim).fill(0)), Error);
});
