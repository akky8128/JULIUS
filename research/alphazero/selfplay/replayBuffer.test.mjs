/**
 * replayBuffer.test.mjs — P4: リングバッファの窓退避・一様サンプリング・resume。
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  openBuffer,
  appendShard,
  sampleToFile,
  bufferStats,
  countJsonlLines,
  saveManifest,
} from "./replayBuffer.mjs";

function tmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "azbuf-"));
}
function writeShard(dir, name, n, tag) {
  const p = path.join(dir, name);
  const lines = [];
  for (let i = 0; i < n; i++) lines.push(JSON.stringify({ tag, i }));
  fs.writeFileSync(p, lines.join("\n") + "\n");
  return p;
}

test("countJsonlLines: 末尾改行の有無に頑健", () => {
  const d = tmpDir();
  fs.writeFileSync(path.join(d, "a.jsonl"), "x\ny\nz\n");
  fs.writeFileSync(path.join(d, "b.jsonl"), "x\ny\nz");
  assert.equal(countJsonlLines(path.join(d, "a.jsonl")), 3);
  assert.equal(countJsonlLines(path.join(d, "b.jsonl")), 3);
});

test("窓退避: capacity 超過で古いシャードが脱落する", () => {
  const src = tmpDir();
  const bufDir = tmpDir();
  const buf = openBuffer(bufDir, { capacityStates: 25 });
  // 各10状態のシャードを4つ追加 → 合計40 > 25 → 古いものが退避される
  for (let k = 0; k < 4; k++) {
    appendShard(buf, writeShard(src, `s${k}.jsonl`, 10, `t${k}`), { round: k });
  }
  const stats = bufferStats(buf);
  assert.ok(stats.totalStates <= 25, `保持状態数は capacity 以下: ${stats.totalStates}`);
  // 最古(t0)は退避されている（保持シャードの round は新しい側）
  const rounds = buf.manifest.shards.map((s) => s.round);
  assert.ok(!rounds.includes(0), "最古 round=0 のシャードは退避されているべき");
  assert.ok(rounds.includes(3), "最新 round=3 は保持されているべき");
});

test("退避: archiveDir 指定時は削除でなく移動される", () => {
  const src = tmpDir();
  const bufDir = tmpDir();
  const arc = tmpDir();
  const buf = openBuffer(bufDir, { capacityStates: 15, archiveDir: arc });
  appendShard(buf, writeShard(src, "a.jsonl", 10, "a"), { round: 1 });
  appendShard(buf, writeShard(src, "b.jsonl", 10, "b"), { round: 2 }); // 20>15 → 最古退避
  const archived = fs.readdirSync(arc);
  assert.ok(archived.length >= 1, "退避シャードが archiveDir に移動されているべき");
});

test("一様サンプリング: 保持シャードから n 件を JSONL 書き出し（seed固定で再現）", () => {
  const src = tmpDir();
  const bufDir = tmpDir();
  const buf = openBuffer(bufDir, { capacityStates: 1000 });
  appendShard(buf, writeShard(src, "a.jsonl", 50, "a"));
  appendShard(buf, writeShard(src, "b.jsonl", 50, "b"));
  const out1 = path.join(bufDir, "batch1.jsonl");
  const out2 = path.join(bufDir, "batch2.jsonl");
  const mul = (s) => () => { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const r1 = sampleToFile(buf, 20, out1, mul(42));
  const r2 = sampleToFile(buf, 20, out2, mul(42));
  assert.equal(r1.sampled, 20);
  assert.equal(r1.seen, 100);
  // 同一 seed → 同一サンプル
  assert.equal(fs.readFileSync(out1, "utf8"), fs.readFileSync(out2, "utf8"));
  assert.equal(countJsonlLines(out1), 20);
});

test("サンプリング: n が総状態数を超える場合は全件返す", () => {
  const src = tmpDir();
  const bufDir = tmpDir();
  const buf = openBuffer(bufDir, { capacityStates: 1000 });
  appendShard(buf, writeShard(src, "a.jsonl", 7, "a"));
  const out = path.join(bufDir, "b.jsonl");
  const r = sampleToFile(buf, 100, out, Math.random);
  assert.equal(r.sampled, 7);
  assert.equal(countJsonlLines(out), 7);
});

test("resume: manifest を保存 → 再 openBuffer で状態が復元される", () => {
  const src = tmpDir();
  const bufDir = tmpDir();
  const buf = openBuffer(bufDir, { capacityStates: 1000 });
  buf.manifest.round = 5;
  appendShard(buf, writeShard(src, "a.jsonl", 30, "a"), { round: 5 });
  saveManifest(buf);

  const reopened = openBuffer(bufDir, { capacityStates: 1000 });
  const stats = bufferStats(reopened);
  assert.equal(stats.round, 5);
  assert.equal(stats.totalStates, 30);
  assert.equal(stats.numShards, 1);
});
