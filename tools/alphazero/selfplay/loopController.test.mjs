/**
 * loopController.test.mjs — 継続更新型ループ機構を mock 依存注入で端から端まで検証。
 * ★実計算ゼロ（実 net ロード・実 MCTS・実 train を一切呼ばない）で配線を確認する★
 * gating（採否・champion・2標本z）は廃止済み。毎周 net が無条件advanceすることを検証する。
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { runLoop, loadLoopState } from "./loopController.mjs";

function tmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "azloop-"));
}

/** 実計算ゼロの mock 依存群を作る。呼び出し回数を記録する。 */
function makeMocks() {
  const calls = { selfPlay: 0, append: 0, sample: 0, train: 0 };
  let totalStates = 0;
  return {
    calls,
    deps: {
      selfPlay({ round }) { calls.selfPlay++; return { shardPath: `/mock/shard_${round}.jsonl`, states: 100 }; },
      appendToBuffer() { calls.append++; totalStates += 100; return { totalStates }; },
      sampleBatch({ n }) { calls.sample++; return { batchPath: `/mock/batch.jsonl`, sampled: n }; },
      train({ round }) { calls.train++; return { checkpointPath: `/mock/ckpt_${round}.json` }; },
    },
  };
}

test("runLoop: rounds が非正だと throw（無人・無限ループ禁止）", () => {
  const { deps } = makeMocks();
  assert.throws(() => runLoop(deps, { dir: tmpDir(), rounds: 0, seedNet: "c0" }), /正の整数/);
});

test("★事前フライトガード: 非dryRun・allowRealRun未指定なら selfPlay/train を1回も呼ばず throw", () => {
  const dir = tmpDir();
  const { deps, calls } = makeMocks();
  assert.throws(
    () => runLoop(deps, { dir, rounds: 3, seedNet: "c0", log: () => {} }),
    /allowRealRun/
  );
  assert.equal(calls.selfPlay, 0);
  assert.equal(calls.append, 0);
  assert.equal(calls.sample, 0);
  assert.equal(calls.train, 0);
});

test("各周で selfPlay→append→sample→train が1回ずつ呼ばれ、net が無条件advanceする", () => {
  const dir = tmpDir();
  const { deps, calls } = makeMocks();
  const st = runLoop(deps, { dir, rounds: 3, seedNet: "models/genAZ001.json", allowRealRun: true, log: () => {} });
  assert.equal(calls.selfPlay, 3);
  assert.equal(calls.append, 3);
  assert.equal(calls.sample, 3);
  assert.equal(calls.train, 3);
  assert.equal(st.round, 3);
  assert.equal(st.net, "/mock/ckpt_3.json", "毎周 net が無条件更新され最終周のckptになる");
  // history に各周の advance が記録される
  assert.equal(st.history.length, 3);
  assert.equal(st.history[0].prevNet, "models/genAZ001.json");
  assert.equal(st.history[0].net, "/mock/ckpt_1.json");
});

test("継続更新: seedNet→r1→r2→r3 と単調に置き換わる（champion概念なし）", () => {
  const dir = tmpDir();
  const { deps } = makeMocks();
  const st = runLoop(deps, { dir, rounds: 3, seedNet: "seed", allowRealRun: true, log: () => {} });
  const nets = st.history.map((h) => h.net);
  assert.deepEqual(nets, ["/mock/ckpt_1.json", "/mock/ckpt_2.json", "/mock/ckpt_3.json"]);
});

test("dryRun: selfPlay と append のみ、sample/train は呼ばれない（安全側既定）", () => {
  const dir = tmpDir();
  const { deps, calls } = makeMocks();
  runLoop(deps, { dir, rounds: 2, seedNet: "c0", dryRun: true, log: () => {} });
  assert.equal(calls.selfPlay, 2);
  assert.equal(calls.append, 2);
  assert.equal(calls.train, 0);
  assert.equal(calls.sample, 0);
});

test("cadence: trainEveryRounds=2 は偶数周のみ学習・advance", () => {
  const dir = tmpDir();
  const { deps, calls } = makeMocks();
  const st = runLoop(deps, { dir, rounds: 4, seedNet: "c0", trainEveryRounds: 2, allowRealRun: true, log: () => {} });
  assert.equal(calls.selfPlay, 4, "自己対戦は毎周");
  assert.equal(calls.train, 2, "学習は偶数周のみ");
  assert.equal(st.net, "/mock/ckpt_4.json", "最後に学習した偶数周のckptが最新net");
});

test("resume: 中断後に再 runLoop でラウンドが継続し net が引き継がれる", () => {
  const dir = tmpDir();
  const { deps: d1 } = makeMocks();
  const st1 = runLoop(d1, { dir, rounds: 2, seedNet: "seed", allowRealRun: true, log: () => {} });
  assert.equal(st1.net, "/mock/ckpt_2.json");
  assert.equal(st1.round, 2);

  const { deps: d2, calls: c2 } = makeMocks();
  const st2 = runLoop(d2, { dir, rounds: 2, seedNet: "IGNORED_ON_RESUME", allowRealRun: true, log: () => {} });
  assert.equal(st2.round, 4, "resume でラウンドが継続する");
  assert.equal(st2.net, "/mock/ckpt_4.json", "net は継続更新され続ける");
  assert.equal(c2.selfPlay, 2);

  const persisted = loadLoopState(dir, "x");
  assert.equal(persisted.round, 4);
});

test("旧 gated manifest（champion フィールド）を読んでも net として復元する（後方互換）", () => {
  const dir = tmpDir();
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "loop_manifest.json"),
    JSON.stringify({ round: 5, champion: "models/genAZ_r5.json", acceptStreak: 1, history: [] }));
  const st = loadLoopState(dir, "seed");
  assert.equal(st.net, "models/genAZ_r5.json");
  assert.equal(st.round, 5);
});
