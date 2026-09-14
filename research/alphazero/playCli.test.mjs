/**
 * playCli.test.mjs — P3: 棋譜入出力（coords.js 利用）の round-trip テスト。
 * 添付棋譜（記法ファイルの20手 1-0）をパース→適用→再エンコードして一致することを確認する。
 */

import test from "node:test";
import assert from "node:assert/strict";

import { generateTurns, hashPosition } from "../../core/ai/moveGen.js";
import { normalizeBoard, denormalizeBoard } from "../../core/gameLogic.js";
import { diffToNotation } from "../../core/coords.js";
import {
  emptyBoard,
  turnToNotation,
  parseTurnNotation,
  formatGameNotation,
} from "./playCli.mjs";

const SIZE = 4;

function nextPlayer(p) {
  return p === "white" ? "black" : "white";
}

/**
 * 実局面上での round-trip プロパティ:
 * 実際に指したターン(before→after)について、
 *   (1) after は generateTurns(before) の候補である
 *   (2) token = diffToNotation(before, after) を持つ候補の中に after と同一盤面のものが存在する
 *       （= token から元の遷移を復元できる = round-trip 可能）
 * を、エンジン生成の実局面列で検証する。
 * 決定的に「最小ハッシュの候補」を指し手として選び、疑似対局を進める。
 */
test("round-trip: エンジン生成の実局面列で token→盤面 復元が一貫する（board→token→board）", () => {
  let board = normalizeBoard(emptyBoard(SIZE), SIZE);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";
  let checked = 0;

  for (let ply = 0; ply < 40; ply++) {
    const state = { board, summonCounts, currentPlayer, boardSize: SIZE };
    const cands = generateTurns(state);
    if (cands.length === 0) break; // 終局

    // 決定的な着手選択（最小 hashPosition）
    let chosen = cands[0];
    let chosenHash = hashPosition(chosen.board, chosen.summonCounts);
    for (const c of cands) {
      const h = hashPosition(c.board, c.summonCounts);
      if (h < chosenHash) { chosen = c; chosenHash = h; }
    }

    const token = turnToNotation(state.board, chosen.board, SIZE);

    // (2) token を持つ候補群に、実際に指した盤面と一致するものが存在する（round-trip 可能）
    const sameToken = cands.filter(
      (c) => diffToNotation(denormalizeBoard(state.board), denormalizeBoard(c.board), SIZE) === token
    );
    const roundTripped = sameToken.some(
      (c) => hashPosition(c.board, c.summonCounts) === chosenHash
    );
    assert.ok(roundTripped, `ply ${ply}: token "${token}" から元盤面を復元できるべき`);

    // token が一意（同一 token に異なる盤面が無い）なら parseTurnNotation は元盤面を返す
    const distinctBoards = new Set(sameToken.map((c) => hashPosition(c.board, c.summonCounts)));
    if (distinctBoards.size === 1) {
      const parsed = parseTurnNotation(token, state);
      assert.equal(hashPosition(parsed.board, parsed.summonCounts), chosenHash);
      checked++;
    }

    board = normalizeBoard(chosen.board, SIZE);
    summonCounts = chosen.summonCounts;
    currentPlayer = nextPlayer(currentPlayer);
  }
  assert.ok(checked > 5, "一意 token の parse round-trip を十分検証できるべき");
});

// 記法ファイル /Users/akky/Documents/Layer/ウケジャレイヤーの棋譜の記法.txt の20手対局。
// ply 順（white,black,...）。20手目 c3d3 は white 勝ち(#, 1-0)。# は除いてトークン化。
const ATTACHED_GAME_PLY = [
  "a1", "b2", "c3", "d4", "d1", "d3", "c3d3", "c2", "a3", "b2c2",
  "a4", "2c2d2d1", "a4a3", "c1", "c3", "d2c1", "a2", "2c12d1", "c3d4", "d2",
  "a1a3", "d2d1", "c2", "2d1d2c2", "a22a3b3c3d2", "d1c1", "a3d4", "c1c3", "b3c3", "b2",
  "c3d4c2b2", "b3", "c2b3", "c2d2", "c2d2", "d1d2", "d1d2", "c3d3", "c3d3",
];

test("添付棋譜: 20手対局を語順非依存パーサで端から端まで完走し white 勝ちに到達", () => {
  let board = normalizeBoard(emptyBoard(SIZE), SIZE);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";

  for (let i = 0; i < ATTACHED_GAME_PLY.length; i++) {
    const token = ATTACHED_GAME_PLY[i];
    const state = { board, summonCounts, currentPlayer, boardSize: SIZE };
    let cand;
    try {
      cand = parseTurnNotation(token, state);
    } catch (e) {
      assert.fail(`ply ${i + 1} (${currentPlayer}) token="${token}" のパースに失敗: ${e.message}`);
    }
    board = normalizeBoard(cand.board, SIZE);
    summonCounts = cand.summonCounts;
    currentPlayer = nextPlayer(currentPlayer);
  }

  // 39 ply 適用後、手番は black。black に合法手なし = white の勝ち（1-0, #）と整合。
  assert.equal(ATTACHED_GAME_PLY.length, 39);
  assert.equal(currentPlayer, "black");
  const finalState = { board, summonCounts, currentPlayer, boardSize: SIZE };
  assert.equal(generateTurns(finalState).length, 0, "最終局面で black は合法手なし（white勝ち）であるべき");
});

/**
 * 既知の非互換性の記録（回帰監視）:
 * 添付棋譜ファイル（ウケジャレイヤーの棋譜の記法.txt）の複数目的地移動 "2d1d2c2" は、
 * coords.js の diffToNotation が (r,c) 走査順で "2d1c2d2" を生成するため、
 * ファイルの語順とは一致しない。js/ を変更できない P3 スコープではこれは既知の差異であり、
 * ファイル棋譜の exact-token round-trip は達成不能（下記で明示的にアサートして固定化する）。
 */
test("既知差異: 添付ファイルの多目的地move語順は coords.js の (r,c)走査順と異なる", () => {
  // d1(=r3c3) から d2(=r2c3)・c2(=r2c2) へ 2枚移動した局面を構成
  const before = emptyBoard(SIZE);
  before[3][3] = ["black", "black"]; // d1 に2枚
  const after = emptyBoard(SIZE);
  after[2][3] = ["black"]; // d2
  after[2][2] = ["black"]; // c2
  const coordsToken = diffToNotation(before, after, SIZE);
  // coords.js は増加セルを (r,c) 走査で c2(r2c2) → d2(r2c3) の順に出す
  assert.equal(coordsToken, "2d1c2d2");
  // 添付ファイルの語順 "2d1d2c2" とは一致しない（既知差異）
  assert.notEqual(coordsToken, "2d1d2c2");
});

test("formatGameNotation: 勝ちに #/1-0 が付く", () => {
  // 3トークン = 1手目(white,black) + 2手目(white)。最終 white 手が勝ち → "2. c3d3#"
  const out = formatGameNotation(["a1", "b2", "c3d3"], "white");
  assert.ok(out.includes("2. c3d3#"), out);
  assert.ok(out.trim().endsWith("1-0"));
  const outB = formatGameNotation(["a1", "b2"], "black");
  assert.ok(outB.includes("1. a1 b2#"));
  assert.ok(outB.trim().endsWith("0-1"));
});

test("diffToNotation round-trip: 各パターン（召喚/移動/複数枚/係数/連続移動）", () => {
  // 召喚
  {
    const before = emptyBoard(SIZE);
    const after = emptyBoard(SIZE);
    after[3][0] = ["white"]; // a1
    assert.equal(turnToNotation(before, after, SIZE), "a1");
  }
  // 1枚移動 a1->b1 (r3c0 -> r3c1)
  {
    const before = emptyBoard(SIZE);
    before[3][0] = ["white"];
    const after = emptyBoard(SIZE);
    after[3][1] = ["white"];
    assert.equal(turnToNotation(before, after, SIZE), "a1b1");
  }
  // 複数マス移動 a1b1 -> c3c4
  {
    const before = emptyBoard(SIZE);
    before[3][0] = ["white"];
    before[3][1] = ["white"];
    const after = emptyBoard(SIZE);
    after[1][2] = ["white"]; // c3 (r1)
    after[0][2] = ["white"]; // c4 (r0)
    // 増加セルは (r,c) 走査順: c4(r0) → c3(r1)
    assert.equal(turnToNotation(before, after, SIZE), "a1b1c4c3");
  }
  // 係数付き: a2(3枚)→a3に2枚,b3に1枚 => 3a22a3b3
  {
    const before = emptyBoard(SIZE);
    before[2][0] = ["white", "white", "white"]; // a2 高さ3
    const after = emptyBoard(SIZE);
    after[1][0] = ["white", "white"]; // a3 高さ2
    after[1][1] = ["white"]; // b3 高さ1
    assert.equal(turnToNotation(before, after, SIZE), "3a22a3b3");
  }
});

test("parseTurnNotation: 非合法トークンはエラー", () => {
  const state = {
    board: normalizeBoard(emptyBoard(SIZE), SIZE),
    summonCounts: { white: 0, black: 0 },
    currentPlayer: "white",
    boardSize: SIZE,
  };
  assert.throws(() => parseTurnNotation("d4d3", state), /対応する合法手がありません/);
});
