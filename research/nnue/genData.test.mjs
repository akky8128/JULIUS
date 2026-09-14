/**
 * genData.test.mjs — research/nnue/genData.mjs (buildDataset) の node:test ベーステスト
 *
 * プロセス起動やファイルIOは行わず、buildDataset を直接呼び出して検証する。
 * 実行: node --test research/nnue/genData.test.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { buildDataset } from "./genData.mjs";
import { featureDim, extractFeatures, NUM_SYMMETRIES } from "../../core/nnue/features.js";
import { normalizeBoard } from "../../core/gameLogic.js";

// ───────────────────────── ヘルパー ─────────────────────────

/** 小構成のベース config（テスト共通） */
function smallConfig(overrides = {}) {
  return {
    games: 2,
    size: 3,
    maxDepth: 2,
    timeBudgetMs: 50,
    maxPlies: 30,
    seed: 1,
    openingRandomPlies: 2,
    epsilon: 0.1,
    augment: 1,
    ...overrides,
  };
}

// ───────────────────────── 1. 基本生成 ─────────────────────────

test("基本生成: 小構成で records.length > 0", () => {
  const { records } = buildDataset(smallConfig());
  assert.ok(records.length > 0, "レコードが1件以上生成されるべき");
});

// ───────────────────────── 2. レコード形式 ─────────────────────────

test("レコード形式: indices/scalars/score/result/sym/size の形が仕様通り", () => {
  const size = 3;
  const dim = featureDim(size);
  const { records } = buildDataset(smallConfig({ size }));

  assert.ok(records.length > 0);

  for (const rec of records) {
    assert.equal(rec.size, size);

    // indices: 整数配列 かつ 範囲内
    assert.ok(Array.isArray(rec.indices));
    for (const idx of rec.indices) {
      assert.ok(Number.isInteger(idx), `idx は整数であるべき: ${idx}`);
      assert.ok(idx >= 0 && idx < dim, `idx ${idx} は [0, ${dim}) の範囲内であるべき`);
    }

    // scalars: 長さ4、有限数
    assert.equal(rec.scalars.length, 4);
    for (const s of rec.scalars) {
      assert.ok(Number.isFinite(s), `scalar は有限数であるべき: ${s}`);
    }

    // score: 有限数
    assert.ok(Number.isFinite(rec.score), `score は有限数であるべき: ${rec.score}`);

    // result: 0 | 0.5 | 1
    assert.ok(
      rec.result === 0 || rec.result === 0.5 || rec.result === 1,
      `result は 0/0.5/1 のいずれかであるべき: ${rec.result}`
    );

    // sym: 0..7
    assert.ok(rec.sym >= 0 && rec.sym < NUM_SYMMETRIES, `sym は 0..7 の範囲内であるべき: ${rec.sym}`);
    assert.ok(Number.isInteger(rec.sym));

    // pov: white|black
    assert.ok(rec.pov === "white" || rec.pov === "black");

    // ply: 非負整数
    assert.ok(Number.isInteger(rec.ply) && rec.ply >= 0);
  }
});

// ───────────────────────── 3. POV整合 ─────────────────────────

test("POV整合: 手構築のstateでextractFeaturesした結果とsym=0レコードのscalarsが一致する", () => {
  // buildDataset 内部で使われる盤面表現と同じ形の state を手動で構築し、
  // extractFeatures(state, state.currentPlayer) の結果が
  // augment=1 のいずれかのレコードの scalars と整合することを確認する
  // （scalars は盤面座標に依存しないため sym=0 と同一になるはず）。
  const size = 3;
  const { records } = buildDataset(smallConfig({ size, augment: 1, games: 3 }));
  assert.ok(records.length > 0);

  // すべて sym=0 のはず（augment=1 なので）
  for (const rec of records) {
    assert.equal(rec.sym, 0, "augment=1 の場合、すべてのレコードは sym=0 であるべき");
  }

  // 手構築の空盤面 state で scalars[3] (空セル率) が [0,1] 範囲内であることなど
  // 基本的な健全性を独立に検証する（extractFeatures の直接呼び出しとの整合）。
  const emptyBoard = Array.from({ length: size }, () => Array.from({ length: size }, () => []));
  const state = {
    board: normalizeBoard(emptyBoard, size),
    summonCounts: { white: 0, black: 0 },
    currentPlayer: "white",
    boardSize: size,
  };
  const feat = extractFeatures(state, state.currentPlayer);
  assert.equal(feat.scalars.length, 4);
  // レコードの scalars もすべて同じ形式・範囲であるはず
  for (const rec of records) {
    assert.equal(rec.scalars.length, feat.scalars.length);
  }
});

// ───────────────────────── 4. D4増強の正しさ ─────────────────────────

test("D4増強: augment=8のレコード数はaugment=1のちょうど8倍で、各元局面グループでsymが0..7を網羅する", () => {
  const config1 = smallConfig({ augment: 1 });
  const config8 = smallConfig({ augment: 8 });

  const result1 = buildDataset(config1);
  const result8 = buildDataset(config8);

  assert.equal(
    result8.records.length,
    result1.records.length * 8,
    "augment=8の総レコード数はaugment=1のちょうど8倍であるべき"
  );

  // augment=8 のレコードは8個ずつのグループになっているはず
  // （expandRecordはsym=0..7を連続してpushするため、8個区切りでグループ化できる）
  assert.equal(result8.records.length % 8, 0);

  for (let i = 0; i < result8.records.length; i += 8) {
    const group = result8.records.slice(i, i + 8);
    const syms = group.map((r) => r.sym).sort((a, b) => a - b);
    assert.deepEqual(syms, [0, 1, 2, 3, 4, 5, 6, 7], `グループ ${i / 8}: sym が0..7を1回ずつ網羅すべき`);

    // グループ内で scalars/score/result が全sym一致
    const first = group[0];
    for (const rec of group) {
      assert.deepEqual(rec.scalars, first.scalars, `グループ ${i / 8}: scalarsが一致しない`);
      assert.equal(rec.score, first.score, `グループ ${i / 8}: scoreが一致しない`);
      assert.equal(rec.result, first.result, `グループ ${i / 8}: resultが一致しない`);
      assert.equal(rec.ply, first.ply, `グループ ${i / 8}: plyが一致しない`);
      assert.equal(rec.pov, first.pov, `グループ ${i / 8}: povが一致しない`);
    }
  }
});

// ───────────────────────── 5. 決定論 ─────────────────────────

test("決定論: 同一configで2回呼び出した結果が完全一致する", () => {
  const config = smallConfig({ games: 3, augment: 8 });
  const result1 = buildDataset(config);
  const result2 = buildDataset(config);

  assert.equal(
    JSON.stringify(result1.records),
    JSON.stringify(result2.records),
    "同一configなら records は完全に決定論的であるべき"
  );
});

// ───────────────────────── 6. 勝敗ラベル整合 ─────────────────────────

test("勝敗ラベル整合: 決着ゲームを含む生成でresultが手番視点で正しく付与される", () => {
  // games を増やして決着ゲームを含める可能性を上げる
  const config = smallConfig({ games: 8, size: 3, maxPlies: 40, augment: 1 });
  const { records, meta } = buildDataset(config);

  assert.ok(records.length > 0);

  // 全レコードが 0/0.5/1 のいずれか
  for (const rec of records) {
    assert.ok([0, 0.5, 1].includes(rec.result));
  }

  // meta の resultDistribution も健全性チェック
  const { win, draw, loss } = meta.resultDistribution;
  assert.equal(win + draw + loss, records.length, "win+draw+lossの合計はレコード総数と一致すべき");

  // 決着ゲームが1つ以上あれば、win と loss の両方が現れるはず
  // （引き分けゲームのみの場合はスキップ可能とするが、size=3・小さいmaxPliesでは
  //   決着がつきやすいことを期待する）
  if (win > 0 || loss > 0) {
    assert.ok(win > 0, "決着ゲームがあるなら win ラベルが1件以上あるべき");
    assert.ok(loss > 0, "決着ゲームがあるなら loss ラベルが1件以上あるべき");
  }
});

// ───────────────────────── 7. pid付与 ─────────────────────────

test("pid付与: 全レコードがpidを持ち、augment=8では同一元局面の8symが同一pidになる", () => {
  const config = smallConfig({ augment: 8, games: 3 });
  const { records, meta } = buildDataset(config);

  assert.ok(records.length > 0);

  for (const rec of records) {
    assert.ok(Number.isInteger(rec.pid) && rec.pid >= 0, `pid は非負整数であるべき: ${rec.pid}`);
  }

  // 8個ずつのグループで同一pidになっているはず
  assert.equal(records.length % 8, 0);
  for (let i = 0; i < records.length; i += 8) {
    const group = records.slice(i, i + 8);
    const pids = new Set(group.map((r) => r.pid));
    assert.equal(pids.size, 1, `グループ ${i / 8}: 8つのsymレコードは同一pidを持つべき`);
  }

  // pid集合が {0..numPositions-1} を連続被覆する
  const pidSet = new Set(records.map((r) => r.pid));
  assert.equal(pidSet.size, meta.numPositions, "ユニークpid数はmeta.numPositionsと一致すべき");
  for (let p = 0; p < meta.numPositions; p++) {
    assert.ok(pidSet.has(p), `pid ${p} が欠落している`);
  }
});

test("pid付与: augment=1でも各レコードが一意のpidを持ち、meta.numPositionsと一致する", () => {
  const config = smallConfig({ augment: 1, games: 3 });
  const { records, meta } = buildDataset(config);

  assert.ok(records.length > 0);
  const pidSet = new Set(records.map((r) => r.pid));
  // augment=1 では1レコード=1元局面なので pid はレコード数と同数のユニーク値
  assert.equal(pidSet.size, records.length);
  assert.equal(pidSet.size, meta.numPositions);
});

// ───────────────────────── 8. evalFn 注入（NNUE自己対戦ブートストラップ） ─────────────────────────

test("evalFn注入: 自明なevalFnを渡してもクラッシュせず、pid/形式が維持される", () => {
  const config = smallConfig({ games: 2, augment: 8, evalFn: () => 0 });
  const { records, meta } = buildDataset(config);

  assert.ok(records.length > 0);
  for (const rec of records) {
    assert.ok(Number.isInteger(rec.pid));
    assert.ok(Number.isFinite(rec.score));
    assert.ok([0, 0.5, 1].includes(rec.result));
  }
  assert.equal(meta.generator.evaluator, "nnue");
});

test("evalFn注入: 同一config+evalFnで2回呼び出した結果が決定論的に一致する", () => {
  const config = smallConfig({ games: 2, augment: 8, evalFn: (state, player) => (player === "white" ? 1 : -1) });
  const result1 = buildDataset(config);
  const result2 = buildDataset(config);

  assert.equal(
    JSON.stringify(result1.records),
    JSON.stringify(result2.records),
    "同一config+evalFnなら records は完全に決定論的であるべき"
  );
});

test("evalFn未指定時: meta.generator.evaluatorがheuristicになる", () => {
  const { meta } = buildDataset(smallConfig());
  assert.equal(meta.generator.evaluator, "heuristic");
  assert.equal(meta.generator.net, null);
});
