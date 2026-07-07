/**
 * tutorial.test.mjs — チュートリアル基盤の node:test ベーステスト
 *
 * 実行: node --test tools/tutorial.test.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { parseCoord, formatCoord, diffToNotation } from "../js/tutorial/coords.js";
import {
  matchesExpectation,
  buildSummonAction,
  buildMoveAction,
  buildEliminateAction,
  tryApplyActions,
  legalDestinations,
  maxMovableCount,
  TutorialEngine,
} from "../js/tutorial/engine.js";
import { checkWinCondition, maxSummonsFor } from "../js/gameLogic.js";
import { scenarios } from "../js/tutorial/scenarios.js";

const BOARD_SIZE = 4;

function emptyBoard(size = BOARD_SIZE) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
}

// ───────────────────────── parseCoord / formatCoord ─────────────────────────

test("parseCoord: a1 は左下 = {r: boardSize-1, c: 0}", () => {
  assert.deepEqual(parseCoord("a1", BOARD_SIZE), { r: 3, c: 0 });
});

test("parseCoord: a4 は左上 = {r: 0, c: 0}", () => {
  assert.deepEqual(parseCoord("a4", BOARD_SIZE), { r: 0, c: 0 });
});

test("parseCoord: d1 は右下 = {r: boardSize-1, c: 3}", () => {
  assert.deepEqual(parseCoord("d1", BOARD_SIZE), { r: 3, c: 3 });
});

test("formatCoord: parseCoordとの往復が一致する", () => {
  const cases = ["a1", "a4", "d1", "b2", "c3"];
  for (const coord of cases) {
    const { r, c } = parseCoord(coord, BOARD_SIZE);
    assert.equal(formatCoord(r, c, BOARD_SIZE), coord, `failed round-trip for ${coord}`);
  }
});

test("formatCoord → parseCoord の往復（全マス）", () => {
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const coord = formatCoord(r, c, BOARD_SIZE);
      assert.deepEqual(parseCoord(coord, BOARD_SIZE), { r, c });
    }
  }
});

// ───────────────────────── diffToNotation ─────────────────────────

test("diffToNotation: 召喚 c3 単独 → 'c3'", () => {
  const before = emptyBoard();
  const after = emptyBoard();
  const { r, c } = parseCoord("c3", BOARD_SIZE);
  after[r][c] = ["white"];

  assert.equal(diffToNotation(before, after, BOARD_SIZE), "c3");
});

test("diffToNotation: 移動 a1,b1 → c3,c4 の盤面差分 → 'a1b1c4c3'", () => {
  // diffToNotationは盤面を行(r)昇順・列(c)昇順で走査してトークン化する。
  // c4はc3より上の行(r=0)にあるため、走査順ではc4が先に現れる。
  const before = emptyBoard();
  const a1 = parseCoord("a1", BOARD_SIZE);
  const b1 = parseCoord("b1", BOARD_SIZE);
  before[a1.r][a1.c] = ["white"];
  before[b1.r][b1.c] = ["white"];

  const after = emptyBoard();
  const c3 = parseCoord("c3", BOARD_SIZE);
  const c4 = parseCoord("c4", BOARD_SIZE);
  after[c3.r][c3.c] = ["white"];
  after[c4.r][c4.c] = ["white"];

  assert.equal(diffToNotation(before, after, BOARD_SIZE), "a1b1c4c3");
});

test("diffToNotation: 同一マス2枚以上変化で係数が付く（3a2 の一部→2a3+b3 で '3a22a3b3'）", () => {
  const before = emptyBoard();
  const a2 = parseCoord("a2", BOARD_SIZE);
  before[a2.r][a2.c] = ["white", "white", "white"];

  const after = emptyBoard();
  const a3 = parseCoord("a3", BOARD_SIZE);
  const b3 = parseCoord("b3", BOARD_SIZE);
  after[a3.r][a3.c] = ["white", "white"];
  after[b3.r][b3.c] = ["white"];

  assert.equal(diffToNotation(before, after, BOARD_SIZE), "3a22a3b3");
});

// ───────────────────────── matchesExpectation ─────────────────────────

test("matchesExpectation: 正しい summon アクションで true", () => {
  const target = parseCoord("c3", BOARD_SIZE);
  const actions = [buildSummonAction(target.r, target.c)];
  assert.equal(matchesExpectation(actions, { type: "summon", at: "c3" }, BOARD_SIZE), true);
});

test("matchesExpectation: 違う summon 位置で false", () => {
  const target = parseCoord("d4", BOARD_SIZE);
  const actions = [buildSummonAction(target.r, target.c)];
  assert.equal(matchesExpectation(actions, { type: "summon", at: "c3" }, BOARD_SIZE), false);
});

test("matchesExpectation: 正しい move アクションで true", () => {
  const from = parseCoord("b2", BOARD_SIZE);
  const to = parseCoord("b3", BOARD_SIZE);
  const actions = [buildMoveAction(from, to, 1)];
  assert.equal(matchesExpectation(actions, { type: "move", from: "b2", to: "b3" }, BOARD_SIZE), true);
});

test("matchesExpectation: 違う move 先で false", () => {
  const from = parseCoord("b2", BOARD_SIZE);
  const to = parseCoord("a2", BOARD_SIZE);
  const actions = [buildMoveAction(from, to, 1)];
  assert.equal(matchesExpectation(actions, { type: "move", from: "b2", to: "b3" }, BOARD_SIZE), false);
});

test("matchesExpectation: type:'any' は何らかのアクションがあれば true", () => {
  const actions = [buildSummonAction(0, 0)];
  assert.equal(matchesExpectation(actions, { type: "any" }, BOARD_SIZE), true);
});

test("matchesExpectation: moveChain は各ステップが一致すれば true / ステップ数違いは false", () => {
  const a3 = parseCoord("a3", BOARD_SIZE);
  const b3 = parseCoord("b3", BOARD_SIZE);
  const c3 = parseCoord("c3", BOARD_SIZE);
  const expect = { type: "moveChain", steps: [{ from: "a3", to: "b3" }, { from: "b3", to: "c3" }] };

  const correct = [buildMoveAction(a3, b3, 2), buildMoveAction(b3, c3, 1)];
  assert.equal(matchesExpectation(correct, expect, BOARD_SIZE), true);

  // 途中で止めた（ステップ数不足）→ false
  const tooShort = [buildMoveAction(a3, b3, 2)];
  assert.equal(matchesExpectation(tooShort, expect, BOARD_SIZE), false);

  // 経路違い → false
  const wrongPath = [buildMoveAction(a3, b3, 2), buildMoveAction(b3, parseCoord("b2", BOARD_SIZE), 1)];
  assert.equal(matchesExpectation(wrongPath, expect, BOARD_SIZE), false);
});

// ───────────────────────── engine + simulateTurn 統合 ─────────────────────────

test("engine: ch2-action-1シナリオでb2への召喚がsimulateTurnを通り棋譜'b2'を生成する", () => {
  const scenario = scenarios.find((s) => s.id === "ch2-action-1");
  assert.ok(scenario, "ch2-action-1 scenario should exist");

  const target = parseCoord("b2", scenario.boardSize);
  const actions = [buildSummonAction(target.r, target.c)];

  const result = tryApplyActions(scenario, actions);
  assert.equal(result.ok, true);
  assert.equal(result.notation, "b2");
  assert.equal(result.summonCounts.white, 1);
  assert.equal(matchesExpectation(actions, scenario.expect, scenario.boardSize), true);
});

test("engine: ch3-action-1シナリオでb2→b3の移動がsimulateTurnを通り棋譜'b2b3'を生成する", () => {
  const scenario = scenarios.find((s) => s.id === "ch3-action-1");
  assert.ok(scenario, "ch3-action-1 scenario should exist");

  const from = parseCoord("b2", scenario.boardSize);
  const to = parseCoord("b3", scenario.boardSize);
  const actions = [buildMoveAction(from, to, 1)];

  const result = tryApplyActions(scenario, actions);
  assert.equal(result.ok, true);
  assert.equal(result.notation, "b2b3");
  assert.equal(matchesExpectation(actions, scenario.expect, scenario.boardSize), true);
});

test("engine: 不正な召喚（既に駒があるマス）はtryApplyActionsがok:falseを返す", () => {
  const scenario = scenarios.find((s) => s.id === "ch3-action-1");
  const occupied = parseCoord("b2", scenario.boardSize); // 既に白駒がある
  const actions = [buildSummonAction(occupied.r, occupied.c)];

  const result = tryApplyActions(scenario, actions);
  assert.equal(result.ok, false);
  assert.ok(typeof result.error === "string" && result.error.length > 0);
});

// ───────────────────────── legalDestinations ─────────────────────────

test("legalDestinations: 角マスは隣接2マス", () => {
  const dests = legalDestinations(BOARD_SIZE, { r: 0, c: 0 });
  assert.equal(dests.length, 2);
});

test("legalDestinations: 辺マスは隣接3マス", () => {
  const dests = legalDestinations(BOARD_SIZE, { r: 0, c: 1 });
  assert.equal(dests.length, 3);
});

test("legalDestinations: 中央マスは隣接4マス", () => {
  const dests = legalDestinations(BOARD_SIZE, { r: 1, c: 1 });
  assert.equal(dests.length, 4);
});

// ───────────────────────── maxMovableCount ─────────────────────────

test("maxMovableCount: 最上段に連続する自駒3枚 → 3", () => {
  const board = emptyBoard();
  board[1][0] = ["white", "white", "white"];
  assert.equal(maxMovableCount(board, { r: 1, c: 0 }, "white", false), 3);
});

test("maxMovableCount: 連続移動中は-1される", () => {
  const board = emptyBoard();
  board[1][0] = ["white", "white", "white"];
  assert.equal(maxMovableCount(board, { r: 1, c: 0 }, "white", true), 2);
});

test("maxMovableCount: 一番上が相手駒なら0", () => {
  const board = emptyBoard();
  board[1][0] = ["white", "black"];
  assert.equal(maxMovableCount(board, { r: 1, c: 0 }, "white", false), 0);
});

// ───────────────────────── 連続移動フロー ─────────────────────────

test("連続移動: a3(3枚)→b3(2枚)→c3(1枚) は合法でmatchesExpectationも真になる", () => {
  const board = emptyBoard();
  board[1][0] = ["white", "white", "white"];
  const scenario = {
    board,
    summonCounts: { white: 8, black: 8 },
    currentPlayer: "white",
    boardSize: BOARD_SIZE,
    expect: { type: "moveChain", steps: [{ from: "a3", to: "b3" }, { from: "b3", to: "c3" }] },
  };
  const a3 = parseCoord("a3", BOARD_SIZE);
  const b3 = parseCoord("b3", BOARD_SIZE);
  const c3 = parseCoord("c3", BOARD_SIZE);
  const actions = [buildMoveAction(a3, b3, 2), buildMoveAction(b3, c3, 1)];

  const result = tryApplyActions(scenario, actions);
  assert.equal(result.ok, true);
  assert.equal(matchesExpectation(actions, scenario.expect, scenario.boardSize), true);
});

test("連続移動: 自駒を残さない継続移動はok:falseになる", () => {
  const board = emptyBoard();
  board[1][0] = ["white", "white", "white"];
  const scenario = {
    board,
    summonCounts: { white: 8, black: 8 },
    currentPlayer: "white",
    boardSize: BOARD_SIZE,
  };
  const a3 = parseCoord("a3", BOARD_SIZE);
  const b3 = parseCoord("b3", BOARD_SIZE);
  const c3 = parseCoord("c3", BOARD_SIZE);
  const actions = [buildMoveAction(a3, b3, 3), buildMoveAction(b3, c3, 3)];

  const result = tryApplyActions(scenario, actions);
  assert.equal(result.ok, false);
});

// ───────────────────────── 章1（座標タップ） ─────────────────────────

test("章1タップ: 3問すべて expect.type==='tap' で、2回誤答までヒントを隠す設定", () => {
  const ids = ["ch1-action-1", "ch1-action-2", "ch1-action-3"];
  const coords = ["c3", "d2", "a4"];
  ids.forEach((id, i) => {
    const s = scenarios.find((sc) => sc.id === id);
    assert.ok(s, `${id} が存在するはず`);
    assert.equal(s.expect.type, "tap");
    assert.equal(s.expect.at, coords[i]);
    assert.equal(s.revealHighlightAfterWrong, 2);
  });
});

test("TutorialEngine: タップステップは正解タップでcanProceedが真になり自動遷移しない", () => {
  const engine = new TutorialEngine({ boardEl: {}, scenarios });
  engine._renderCurrentStep = () => {};

  engine.stepIndex = scenarios.findIndex((s) => s.id === "ch1-action-1");
  engine._resetStepState();
  const idx = engine.stepIndex;

  const c3 = parseCoord("c3", engine.currentScenario.boardSize);
  engine._handleCellClick(c3.r, c3.c);

  assert.equal(engine.tapCleared, true);
  assert.equal(engine.canProceed, true);
  assert.equal(engine.stepIndex, idx, "タップ正解では自動遷移しない");

  engine.next();
  assert.equal(engine.stepIndex, idx + 1, "next()で次へ進む");
});

test("TutorialEngine: 誤タップは wrongCount を増やし、2回でヒントが見えるようになる", () => {
  const engine = new TutorialEngine({ boardEl: {}, scenarios });
  engine._renderCurrentStep = () => {};

  engine.stepIndex = scenarios.findIndex((s) => s.id === "ch1-action-1");
  engine._resetStepState();
  engine.wrongCount = 0;

  // 正解は c3。まずは違うマス（a1）をタップ。
  const a1 = parseCoord("a1", BOARD_SIZE);
  assert.deepEqual(engine._visibleHighlights(), [], "最初はヒントを出さない");
  engine._handleCellClick(a1.r, a1.c);
  assert.equal(engine.wrongCount, 1);
  assert.deepEqual(engine._visibleHighlights(), [], "1回目の誤答ではまだヒントを出さない");
  engine._handleCellClick(a1.r, a1.c);
  assert.equal(engine.wrongCount, 2);
  assert.deepEqual(engine._visibleHighlights(), ["c3"], "2回誤答でヒント（c3）を表示する");
});

// ───────────────────────── 章2〜6 回帰テスト ─────────────────────────

test("章2〜6: 各actionシナリオの正解手がtryApplyActionsでok:trueかつmatchesExpectationが真", () => {
  const chapterCases = [
    {
      id: "ch2-action-1",
      buildActions: (s) => {
        const t = parseCoord("b2", s.boardSize);
        return [buildSummonAction(t.r, t.c)];
      },
      notation: "b2",
    },
    {
      id: "ch3-action-1",
      buildActions: (s) => {
        const from = parseCoord("b2", s.boardSize);
        const to = parseCoord("b3", s.boardSize);
        return [buildMoveAction(from, to, 1)];
      },
      notation: "b2b3",
    },
    {
      id: "ch4-action-1", // 連携: b1→b2→b3→b4（各1枚）で b1b4
      buildActions: (s) => {
        const b1 = parseCoord("b1", s.boardSize);
        const b2 = parseCoord("b2", s.boardSize);
        const b3 = parseCoord("b3", s.boardSize);
        const b4 = parseCoord("b4", s.boardSize);
        return [buildMoveAction(b1, b2, 1), buildMoveAction(b2, b3, 1), buildMoveAction(b3, b4, 1)];
      },
      notation: "b1b4",
    },
    {
      id: "ch4-action-2", // 濃縮: a3(3)→b3(2)→c3(1) で 3a3b3c3d3
      buildActions: (s) => {
        const a3 = parseCoord("a3", s.boardSize);
        const b3 = parseCoord("b3", s.boardSize);
        const c3 = parseCoord("c3", s.boardSize);
        const d3 = parseCoord("d3", s.boardSize);
        return [buildMoveAction(a3, b3, 3), buildMoveAction(b3, c3, 2), buildMoveAction(c3, d3, 1)];
      },
      notation: "3a3b3c3d3",
    },
    {
      id: "ch5-action-1",
      buildActions: (s) => {
        const t = parseCoord("c3", s.boardSize);
        return [buildEliminateAction(t.r, t.c)];
      },
      notation: "c3",
    },
    {
      id: "ch6-action-1",
      buildActions: (s) => {
        const from = parseCoord("d3", s.boardSize);
        const to = parseCoord("d4", s.boardSize);
        return [buildMoveAction(from, to, 1)];
      },
      notation: "d3d4",
    },
  ];

  for (const { id, buildActions, notation } of chapterCases) {
    const scenario = scenarios.find((s) => s.id === id);
    assert.ok(scenario, `${id} scenario should exist`);
    const actions = buildActions(scenario);
    const result = tryApplyActions(scenario, actions);
    assert.equal(result.ok, true, `${id}: expected ok:true, got error: ${result.error}`);
    assert.equal(result.notation, notation, `${id}: 棋譜が期待値と一致するはず`);
    assert.equal(
      matchesExpectation(actions, scenario.expect, scenario.boardSize),
      true,
      `${id}: expected matchesExpectation to be true`
    );
  }
});

test("章6: 正解手を確定した後、checkWinConditionが真になる", () => {
  const scenario = scenarios.find((s) => s.id === "ch6-action-1");
  const from = parseCoord("d3", scenario.boardSize);
  const to = parseCoord("d4", scenario.boardSize);
  const actions = [buildMoveAction(from, to, 1)];

  const result = tryApplyActions(scenario, actions);
  assert.equal(result.ok, true);

  const win = checkWinCondition(
    { board: result.board, summonCounts: result.summonCounts, currentPlayer: "black" },
    { maxSummons: maxSummonsFor(scenario.boardSize), boardSize: scenario.boardSize }
  );
  assert.equal(win, true, "白がd4に乗った後、黒の勝利条件（＝白の勝ち）が成立するはず");
});

// ───────────────────────── info ステップ / next() ─────────────────────────

test("TutorialEngine: infoステップはisInfoStepが真で、next()で次のステップへ進む", () => {
  assert.equal(scenarios[0].kind, "info");

  const engine = new TutorialEngine({ boardEl: {}, scenarios });
  engine._renderCurrentStep = () => {}; // DOM描画を無効化し、進行ロジックのみテストする。

  assert.equal(engine.isInfoStep, true);
  const beforeIndex = engine.stepIndex;
  engine.next();
  assert.equal(engine.stepIndex, beforeIndex + 1);
});

// ───────────────────────── 移動/排除モード切り替え ─────────────────────────

test("TutorialEngine: canToggleEliminateModeは許可された章（allowEliminateMode）でのみ真", () => {
  const engine = new TutorialEngine({ boardEl: {}, scenarios });
  engine._renderCurrentStep = () => {};

  const ch2Index = scenarios.findIndex((s) => s.id === "ch2-action-1"); // 排除許可なし
  const ch5Index = scenarios.findIndex((s) => s.id === "ch5-action-1"); // 排除許可あり + 8/8

  engine.stepIndex = ch2Index;
  assert.equal(engine.canToggleEliminateMode, false);

  engine.stepIndex = ch5Index;
  assert.equal(engine.canToggleEliminateMode, true);
});

test("TutorialEngine: 排除モードで自駒最上段をクリック→ターン終了で排除が確定し次章へ進む", () => {
  const engine = new TutorialEngine({ boardEl: {}, scenarios });
  engine._renderCurrentStep = () => {};

  const kifuTokens = [];
  engine.onKifu = (notation) => kifuTokens.push(notation);

  engine.stepIndex = scenarios.findIndex((s) => s.id === "ch5-action-1");
  engine.setActionMode("eliminate");
  assert.equal(engine.actionMode, "eliminate");

  const c3 = parseCoord("c3", engine.currentScenario.boardSize);
  engine._handleCellClick(c3.r, c3.c);

  // クリックだけでは確定せず、ターン終了待ちになる。
  assert.equal(engine.canEndTurn, true);
  assert.deepEqual(kifuTokens, []);

  engine.endTurn();
  assert.deepEqual(kifuTokens, ["c3"]);
  assert.equal(engine.currentScenario.id, "ch6-info-1");
});

test("TutorialEngine: 移動モードのとき自駒最上段クリックは移動元選択になる（排除は起きない）", () => {
  const engine = new TutorialEngine({ boardEl: {}, scenarios });
  engine._renderCurrentStep = () => {};

  const kifuTokens = [];
  engine.onKifu = (notation) => kifuTokens.push(notation);

  engine.stepIndex = scenarios.findIndex((s) => s.id === "ch5-action-1");
  engine.setActionMode("move");
  assert.equal(engine.actionMode, "move");

  const c3 = parseCoord("c3", engine.currentScenario.boardSize);
  engine._handleCellClick(c3.r, c3.c);

  assert.deepEqual(kifuTokens, []);
  assert.deepEqual(engine.selected, { r: c3.r, c: c3.c });
});

test("TutorialEngine: 召喚は即遷移せず、ターン終了で確定して進む", () => {
  const engine = new TutorialEngine({ boardEl: {}, scenarios });
  engine._renderCurrentStep = () => {};

  const kifuTokens = [];
  engine.onKifu = (notation) => kifuTokens.push(notation);

  engine.stepIndex = scenarios.findIndex((s) => s.id === "ch2-action-1");
  engine._resetStepState();
  const idx = engine.stepIndex;

  const b2 = parseCoord("b2", engine.currentScenario.boardSize);
  engine._handleCellClick(b2.r, b2.c);

  // 召喚をタップした時点では確定せず、ターン終了待ち。
  assert.equal(engine.canEndTurn, true);
  assert.equal(engine.stepIndex, idx, "召喚タップだけでは自動遷移しない");

  engine.endTurn();
  assert.deepEqual(kifuTokens, ["b2"]);
  assert.equal(engine.stepIndex, idx + 1);
});
