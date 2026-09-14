/**
 * microActions.test.mjs — P0 プロパティテスト。
 *
 * (a) 既存 generateTurns（js/ai/moveGen.js）による終端盤面集合
 * (b) microActions.mjs のターン文法で到達可能な終端盤面集合
 * が完全一致することを、ランダム局面1000通りで検証する（PLAN.md §3.5 / §7 P0）。
 */

import test from "node:test";
import assert from "node:assert/strict";

import { generateTurns, hashPosition, MAX_CANDIDATES } from "../../js/ai/moveGen.js";
import { maxSummonsFor } from "../../js/gameLogic.js";
import {
  initTurnState,
  legalMicroActions,
  applyMicroAction,
  moveNode,
  summonNode,
  END_TURN_ACTION,
  MAX_CHAIN_LENGTH,
} from "./microActions.mjs";

const BOARD_SIZE = 4;

// ───────────────────────── 再現可能な乱数生成 ─────────────────────────

function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randInt(rng, maxInclusive) {
  return Math.floor(rng() * (maxInclusive + 1));
}

function choice(rng, arr) {
  return arr[Math.floor(rng() * arr.length)];
}

/**
 * ランダムだが妥当な（n_c <= summonCounts[c] <= 8）局面を生成する。
 * 序盤（召喚途中）・中盤（全召喚済み）・終盤（駒減少）をまんべんなくカバーする。
 */
function randomState(rng, phaseHint) {
  const maxSummons = maxSummonsFor(BOARD_SIZE);
  let summonCounts;
  let piecesOnBoard;

  if (phaseHint === "early") {
    // 序盤: 両者とも召喚途中
    summonCounts = { white: randInt(rng, 4), black: randInt(rng, 4) };
    piecesOnBoard = {
      white: randInt(rng, summonCounts.white),
      black: randInt(rng, summonCounts.black),
    };
  } else if (phaseHint === "asym") {
    // 非対称: 片方だけ全召喚済み・他方は途中（summonPhaseOver 境界の検証用）
    const full = choice(rng, ["white", "black"]);
    const partial = full === "white" ? "black" : "white";
    summonCounts = { [full]: maxSummons, [partial]: randInt(rng, maxSummons - 1) };
    piecesOnBoard = {
      white: randInt(rng, summonCounts.white),
      black: randInt(rng, summonCounts.black),
    };
  } else if (phaseHint === "mid") {
    // 中盤: 全召喚済み・駒はまだ多め（4..8）
    summonCounts = { white: maxSummons, black: maxSummons };
    piecesOnBoard = {
      white: 4 + randInt(rng, maxSummons - 4),
      black: 4 + randInt(rng, maxSummons - 4),
    };
  } else {
    // late: 全召喚済みで排除により駒が減った終盤局面（低駒数バイアス 0..max(1, floor(max/3))）
    summonCounts = { white: maxSummons, black: maxSummons };
    const lowCap = Math.max(1, Math.floor(maxSummons / 3));
    piecesOnBoard = {
      white: randInt(rng, lowCap),
      black: randInt(rng, lowCap),
    };
  }

  const board = Array.from({ length: BOARD_SIZE }, () =>
    Array.from({ length: BOARD_SIZE }, () => [])
  );

  for (const color of ["white", "black"]) {
    for (let i = 0; i < piecesOnBoard[color]; i++) {
      const r = randInt(rng, BOARD_SIZE - 1);
      const c = randInt(rng, BOARD_SIZE - 1);
      board[r][c].push(color);
    }
  }

  // スタック内の色順をシャッフル（同色区別不能だが混在パターンを増やす）
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const stack = board[r][c];
      for (let i = stack.length - 1; i > 0; i--) {
        const j = randInt(rng, i);
        [stack[i], stack[j]] = [stack[j], stack[i]];
      }
    }
  }

  const currentPlayer = choice(rng, ["white", "black"]);
  return { board, summonCounts, currentPlayer, boardSize: BOARD_SIZE };
}

// ───────────────────────── microActions によるDFS総当たり ─────────────────────────

/**
 * microActions の文法で到達可能な全終端盤面のハッシュ集合をDFSで列挙する。
 * @returns {{ hashes: Set<string>, stateCount: number, aborted: boolean }}
 */
function enumerateMicroActionTerminals(state, maxStates = 300000) {
  const hashes = new Set();
  let stateCount = 0;
  let aborted = false;

  function dfs(ctx) {
    if (aborted) return;
    stateCount++;
    if (stateCount > maxStates) {
      aborted = true;
      return;
    }
    const legal = legalMicroActions(ctx);
    for (const action of legal) {
      const next = applyMicroAction(ctx, action);
      if (next.done) {
        hashes.add(hashPosition(next.board, next.summonCounts));
      } else {
        dfs(next);
        if (aborted) return;
      }
    }
  }

  dfs(initTurnState(state));
  return { hashes, stateCount, aborted };
}

function refHashes(state) {
  return new Set(generateTurns(state).map((t) => hashPosition(t.board, t.summonCounts)));
}

// ───────────────────────── 固定局面ユニットテスト ─────────────────────────

test("実質パス禁止: ラウンドトリップで盤面が開始局面に戻るとターン終了ノードが禁止される", () => {
  const board = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
  board[0][0] = ["white", "white"];
  board[0][1] = ["white"];
  const state = {
    board,
    summonCounts: { white: 3, black: 0 },
    currentPlayer: "white",
    boardSize: 4,
  };

  let ctx = initTurnState(state);
  // (0,0) -> (0,1) へ1枚移動（dir=3:右）。AUGMENT(A=(0,0),dir=3,B=(0,1)) に入る。
  ctx = applyMicroAction(ctx, moveNode(0, 0, 3, 4));
  assert.equal(ctx.phase, "AUGMENT");

  // (0,1) -> (0,0) へ1枚戻す（dir=2:左）。これで開始局面と完全一致する。
  const backNode = moveNode(0, 1, 2, 4);
  assert.ok(legalMicroActions(ctx).includes(backNode));
  ctx = applyMicroAction(ctx, backNode);
  assert.equal(hashPosition(ctx.board, ctx.summonCounts), hashPosition(board, state.summonCounts));

  const legal = legalMicroActions(ctx);
  assert.ok(!legal.includes(END_TURN_ACTION), "実質パスとなる終了ノードは禁止されるべき");
});

test("召喚/排除の排他性: 空マスは全召喚済みでも召喚不可、自駒トップかつ相手駒なしのマスは排除不可", () => {
  const board = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
  board[0][0] = ["white"]; // 相手駒なし → 排除不可
  const state = {
    board,
    summonCounts: { white: 8, black: 8 }, // 全召喚済み
    currentPlayer: "white",
    boardSize: 4,
  };
  const ctx = initTurnState(state);
  const legal = legalMicroActions(ctx);

  // (0,0) は自駒トップだが直下に相手駒がないので排除ノードは出ない
  assert.ok(!legal.includes(summonNode(0, 0, 4)));
  // 空マス(1,1)は全召喚済みのため召喚ノードは出ない
  assert.ok(!legal.includes(summonNode(1, 1, 4)));
});

test("召喚は空マスかつ召喚回数残ありでのみ合法", () => {
  const board = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
  const state = {
    board,
    summonCounts: { white: 7, black: 0 },
    currentPlayer: "white",
    boardSize: 4,
  };
  const ctx = initTurnState(state);
  const legal = legalMicroActions(ctx);
  assert.ok(legal.includes(summonNode(2, 2, 4)));
});

test("残置制約: CHAINでの再採用はチェーンヘッドに自駒2枚以上必要", () => {
  const board = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
  board[0][0] = ["white"];
  board[0][1] = ["white"]; // 移動先にすでに1枚
  const state = {
    board,
    summonCounts: { white: 2, black: 0 },
    currentPlayer: "white",
    boardSize: 4,
  };
  let ctx = initTurnState(state);
  ctx = applyMicroAction(ctx, moveNode(0, 0, 3, 4)); // A(0,0)->B(0,1)、B=2枚に
  assert.equal(ctx.phase, "AUGMENT");

  // AUGMENT再採用: ownAtHead(A)=0 のため不可
  assert.ok(!legalMicroActions(ctx).includes(moveNode(0, 0, 3, 4)));

  // Bからチェーンへ: ownAtDest(B)=2 なので合法
  const chainNode = moveNode(0, 1, 1, 4); // dir=1: 下へ
  assert.ok(legalMicroActions(ctx).includes(chainNode));
  ctx = applyMicroAction(ctx, chainNode); // CHAIN(H=(0,1),dir=1,C=(1,1)) 、C=1枚のみ
  assert.equal(ctx.phase, "CHAIN");

  // CHAIN再採用にはH(0,1)に自駒2枚以上必要だが、現在1枚のため不可
  assert.ok(!legalMicroActions(ctx).includes(moveNode(0, 1, 1, 4)));
});

test("ホップ上限: depth が MAX_CHAIN_LENGTH に達するとチェーン継続ノードが出ない", () => {
  const board = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
  board[0][0] = ["white", "white", "white"];
  const state = {
    board,
    summonCounts: { white: 3, black: 0 },
    currentPlayer: "white",
    boardSize: 4,
  };
  let ctx = initTurnState(state);
  ctx = applyMicroAction(ctx, moveNode(0, 0, 3, 4)); // A(0,0)->(0,1) 、B=3枚
  assert.equal(ctx.phase, "AUGMENT");

  // depth を人為的に上限へ引き上げ、チェーン継続ノードが提示されないことを確認
  const cappedCtx = { ...ctx, depth: MAX_CHAIN_LENGTH };
  const legal = legalMicroActions(cappedCtx);
  // (0,1)からのチェーン移動ノード（下/左/右）はどれも含まれないはず
  assert.ok(!legal.includes(moveNode(0, 1, 1, 4)));
  assert.ok(!legal.includes(moveNode(0, 1, 2, 4)));
  assert.ok(!legal.includes(moveNode(0, 1, 3, 4)));
  // 再採用（AUGMENT）はdepthに影響されないので合法のまま
  assert.ok(legal.includes(moveNode(0, 0, 3, 4)));
});

test("反復禁止: 同一フルコンテキストキーの再出現はDFSを有限に停止させる", () => {
  const board = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
  board[0][0] = ["white", "white"];
  board[0][1] = ["white"];
  const state = {
    board,
    summonCounts: { white: 3, black: 0 },
    currentPlayer: "white",
    boardSize: 4,
  };
  const { hashes, stateCount, aborted } = enumerateMicroActionTerminals(state, 5000);
  assert.ok(!aborted, "反復禁止が機能していれば少数局面のDFSは容易に終了するはず");
  assert.ok(hashes.size > 0);
  assert.ok(stateCount < 5000);
});

// ───────────────────────── ランダム1000局面の総当たり等価性テスト ─────────────────────────

test("総当たり等価性: ランダム1000局面で generateTurns と microActions の終端盤面集合が一致する", () => {
  const N = 1000;
  const PHASES = ["early", "asym", "mid", "late"];
  const rng = mulberry32(20260713);
  let cappedSkipped = 0;
  let abortedSkipped = 0;
  let checked = 0;
  const stats = {};
  for (const p of PHASES) {
    stats[p] = { count: 0, summonPhaseOver: 0, eliminationLegal: 0, totalPieces: 0, minPieces: Infinity };
  }
  const maxSummons = maxSummonsFor(BOARD_SIZE);

  for (let i = 0; i < N; i++) {
    const phaseHint = PHASES[i % PHASES.length];
    const state = randomState(rng, phaseHint);

    // フェーズ別カバレッジ統計
    const s = stats[phaseHint];
    s.count++;
    const over = state.summonCounts.white === maxSummons && state.summonCounts.black === maxSummons;
    if (over) s.summonPhaseOver++;
    const pieces = state.board.flat().reduce((acc, st) => acc + st.length, 0);
    s.totalPieces += pieces;
    s.minPieces = Math.min(s.minPieces, pieces);
    if (over) {
      const player = state.currentPlayer;
      const hasElim = state.board.flat().some(
        (st) =>
          st.length >= 2 &&
          st[st.length - 1] === player &&
          st.slice(0, -1).some((p) => p !== player)
      );
      if (hasElim) s.eliminationLegal++;
    }

    const a = refHashes(state);
    if (a.size >= MAX_CANDIDATES) {
      cappedSkipped++;
      continue;
    }

    const { hashes: b, aborted } = enumerateMicroActionTerminals(state);
    if (aborted) {
      abortedSkipped++;
      continue;
    }

    checked++;
    if (a.size !== b.size || ![...a].every((h) => b.has(h))) {
      const onlyInRef = [...a].filter((h) => !b.has(h));
      const onlyInMicro = [...b].filter((h) => !a.has(h));
      assert.fail(
        `局面#${i} (${phaseHint}) で不一致。state=${JSON.stringify(state)}\n` +
          `refのみ(${onlyInRef.length}): ${JSON.stringify(onlyInRef.slice(0, 5))}\n` +
          `microのみ(${onlyInMicro.length}): ${JSON.stringify(onlyInMicro.slice(0, 5))}`
      );
    }
  }

  console.log(
    `[P0] checked=${checked}/${N} cappedSkipped=${cappedSkipped} abortedSkipped=${abortedSkipped}`
  );
  for (const p of PHASES) {
    const s = stats[p];
    console.log(
      `[P0] phase=${p} count=${s.count} summonPhaseOver=${s.summonPhaseOver} ` +
        `eliminationLegal=${s.eliminationLegal} avgPieces=${(s.totalPieces / s.count).toFixed(2)} minPieces=${s.minPieces}`
    );
  }
  assert.ok(checked > 0, "少なくとも1局面は比較できるべき");

  // カバレッジ保証: 各フェーズが意図した領域を実際に踏んでいること
  assert.equal(stats.asym.summonPhaseOver, 0, "asym は summonPhaseOver 境界の手前であるべき");
  assert.equal(stats.mid.summonPhaseOver, stats.mid.count, "mid は全召喚済みであるべき");
  assert.equal(stats.late.summonPhaseOver, stats.late.count, "late は全召喚済みであるべき");
  assert.ok(stats.mid.eliminationLegal > 0, "mid で排除解禁局面が生成されるべき");
  assert.ok(
    stats.late.totalPieces / stats.late.count < stats.mid.totalPieces / stats.mid.count,
    "late は mid より駒数が少ないべき（駒減少終盤のカバレッジ）"
  );
});
