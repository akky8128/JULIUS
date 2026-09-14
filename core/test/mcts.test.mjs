/**
 * mcts.test.mjs — P3: PUCT MCTS のテスト。
 * ★符号伝播（backup sign）を最優先で検証する（PLAN §10: ここが壊れると学習が静かに死ぬ）★
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  makeRootNode,
  makeNode,
  childFor,
  expandNode,
  selectAction,
  simulate,
  runMCTS,
  bestActionByVisits,
  rootQ,
  DEFAULT_C_PUCT,
} from "../az/mcts.mjs";
import {
  initTurnState,
  applyMicroAction,
  legalMicroActions,
  moveNode,
  summonNode,
  NUM_MICRO_ACTIONS,
} from "../az/microActions.mjs";
import { normalizeBoard } from "../gameLogic.js";

// value=0（勝率0.5→v=0）、合法手一様の決定的スタブネット。
function makeStubNet(value = 0) {
  return {
    evaluatePolicy() {
      return { value, policyLogits: new Float64Array(NUM_MICRO_ACTIONS) };
    },
  };
}

function emptyBoard() {
  return Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => []));
}

test("★符号(a)+(c): 相手を即詰みにする召喚手のQが+1（終端-1が手番交代で符号反転）", () => {
  // white=7,black=8, 両者とも盤上0枚。white は移動手を持たず、合法手は召喚のみ。
  // どの空マスに召喚(即done)しても white=8 となり、black は TURN_START で合法手ゼロ
  // （召喚8済/駒0）→ 終端(black視点 -1)。white根には符号反転で +1 が伝播する。
  const state = {
    board: emptyBoard(),
    summonCounts: { white: 7, black: 8 },
    currentPlayer: "white",
    boardSize: 4,
  };
  const root = makeRootNode(state);
  // 合法手はすべて召喚ノード（64..79）であること
  assert.ok(root.legal.length > 0 && root.legal.every((a) => a >= 64 && a < 80));

  runMCTS(root, makeStubNet(0), { simulations: 200 });

  // 最善手は召喚で、その子は black の終端ノード（black視点 -1）
  const best = bestActionByVisits(root);
  assert.ok(best >= 64 && best < 80, "最善手は召喚ノードであるべき");
  const child = childFor(root, best);
  assert.equal(child.mover, "black");
  assert.ok(child.isTerminal);
  assert.equal(child.terminalValue, -1);

  // 終端到達訪問は必ず white根視点 +1 を積むので Q は厳密に +1
  assert.ok(root.N[best] > 0);
  assert.equal(rootQ(root, best), 1);
});

test("★符号(b): マイクロ遷移は符号保存・ENDで符号反転（AUGMENT の END で相手終端）", () => {
  // white=8,black=8, white は (0,0) の1枚のみ、black 盤上0枚。
  // white は必ず「1枚移動→END」の2マイクロで black を終端にできる（唯一の勝ち筋）。
  const board = emptyBoard();
  board[0][0] = ["white"];
  const state = {
    board,
    summonCounts: { white: 8, black: 8 },
    currentPlayer: "white",
    boardSize: 4,
  };
  const root = makeRootNode(state);
  const moveA = moveNode(0, 0, 3, 4); // (0,0)→(0,1)（右）

  // 子(AUGMENT)は手番不変（white）＝マイクロ遷移
  const augment = childFor(root, moveA);
  assert.equal(augment.mover, "white");
  assert.equal(augment.isTerminal, false);
  assert.equal(augment.ctx.phase, "AUGMENT");

  runMCTS(root, makeStubNet(0), { simulations: 300 });

  // AUGMENT の唯一の勝ち筋 END は black 終端へ→ white視点で符号反転して +1（厳密）
  const END = 80;
  assert.ok(augment.legal.includes(END));
  const endChild = childFor(augment, END);
  assert.equal(endChild.mover, "black"); // done で手番交代
  assert.ok(endChild.isTerminal);
  assert.equal(endChild.terminalValue, -1);
  assert.ok(augment.N[END] > 0);
  assert.equal(rootQ(augment, END), 1); // END 反転が +1

  // 根の移動手は符号保存（white→white）で正（0 と +1 の平均 → 正）
  assert.ok(rootQ(root, moveA) > 0);
});

test("PUCT選択式: 既知の N,P,Q で argmax が一致する", () => {
  const node = makeNode(initTurnState({
    board: (() => { const b = emptyBoard(); b[0][0] = ["white"]; return b; })(),
    summonCounts: { white: 1, black: 0 },
    currentPlayer: "white",
    boardSize: 4,
  }), "white", 4);
  // legal を人工的に3手に固定して式を検証
  node.legal = [10, 20, 30];
  node.expanded = true;
  node.P = new Float64Array(NUM_MICRO_ACTIONS);
  node.P[10] = 0.2; node.P[20] = 0.5; node.P[30] = 0.3;
  node.N[10] = 1; node.N[20] = 1; node.N[30] = 0;
  node.W[10] = 0.9; node.W[20] = 0.1; node.W[30] = 0;
  node.Nnode = 2;
  const sqrtN = Math.sqrt(2);
  const score = (a) => (node.N[a] > 0 ? node.W[a] / node.N[a] : 0) + DEFAULT_C_PUCT * node.P[a] * sqrtN / (1 + node.N[a]);
  const expected = [10, 20, 30].reduce((best, a) => (score(a) > score(best) ? a : best), 10);
  assert.equal(selectAction(node, DEFAULT_C_PUCT), expected);
});

test("マスク整合: expandNode の P は legalMicroActions と厳密一致（合法のみ非ゼロ）", () => {
  const board = emptyBoard();
  board[0][0] = ["white", "white"];
  board[2][2] = ["black"];
  const root = makeRootNode({
    board,
    summonCounts: { white: 3, black: 2 },
    currentPlayer: "white",
    boardSize: 4,
  });
  expandNode(root, makeStubNet(0));
  const legalSet = new Set(root.legal);
  for (let a = 0; a < NUM_MICRO_ACTIONS; a++) {
    if (legalSet.has(a)) assert.ok(root.P[a] > 0, `合法手 ${a} の P は正であるべき`);
    else assert.equal(root.P[a], 0, `非合法手 ${a} の P は 0 であるべき`);
  }
  // 子生成が legal に従うこと
  simulate(root, makeStubNet(0));
  for (const a of root.children.keys()) assert.ok(legalSet.has(a));
});

test("決定性: 同一局面・固定simulations・Dirichlet OFF で N 配列が完全一致", () => {
  const mkState = () => {
    const board = emptyBoard();
    board[0][0] = ["white"];
    board[1][1] = ["black", "white"];
    board[3][3] = ["white", "white", "black"];
    return { board, summonCounts: { white: 4, black: 3 }, currentPlayer: "white", boardSize: 4 };
  };
  const net = makeStubNet(0.3);
  const r1 = makeRootNode(mkState());
  const r2 = makeRootNode(mkState());
  runMCTS(r1, net, { simulations: 150 });
  runMCTS(r2, net, { simulations: 150 });
  assert.deepEqual(Array.from(r1.N), Array.from(r2.N));
});

// microActions.mjs 契約: 「TURN_START で合法ゼロ = 真の敗北」「AUGMENT/CHAIN で合法ゼロ =
// 実質パス盤面＋反復禁止で全刈られた袋小路（敗北ではない・放棄すべき）」を区別する。
// 既知の袋小路: 下記 state・path[45,60] で CHAIN の legal.length===0 に到達する。
function reachDeadEndCtx() {
  const board = normalizeBoard([
    [[], ["white"], [], []],
    [["white"], ["white"], [], []],
    [[], [], [], ["black"]],
    [[], ["white"], ["white", "white"], ["white", "white", "black"]],
  ], 4);
  const state = { board, summonCounts: { white: 8, black: 8 }, currentPlayer: "black", boardSize: 4 };
  let ctx = initTurnState(state);
  for (const a of [45, 60]) ctx = applyMicroAction(ctx, a);
  return { ctx, state };
}

test("★CRITICAL1(a): AUGMENT/CHAIN の袋小路(合法ゼロ)は terminal でも -1 でもない", () => {
  const { ctx } = reachDeadEndCtx();
  assert.equal(ctx.phase, "CHAIN");
  assert.equal(legalMicroActions(ctx).length, 0, "既知の袋小路（mid-turn 合法ゼロ）であること");
  const node = makeNode(ctx, "black", 4);
  assert.equal(node.isTerminal, false, "mid-turn 袋小路は terminal ではない（真の敗北ではない）");
  assert.notEqual(node.terminalValue, -1, "袋小路に虚偽の -1 を与えてはならない");
});

test("★CRITICAL1(b): MCTS木に mid-turn 袋小路の isTerminal が混入せず -1 汚染がない", () => {
  const { state } = reachDeadEndCtx();
  const root = makeRootNode(state);
  runMCTS(root, makeStubNet(0), { simulations: 400 });
  // 木を走査し、isTerminal ノードは全て TURN_START（真の終端）であることを確認
  const stack = [root];
  const seen = new Set();
  let terminals = 0;
  let midTurnDeadEndsAsChildren = 0;
  while (stack.length) {
    const n = stack.pop();
    if (seen.has(n)) continue;
    seen.add(n);
    if (n.isTerminal) {
      terminals++;
      assert.equal(n.ctx.phase, "TURN_START", "isTerminal は真の終端(TURN_START 合法ゼロ)のみであるべき");
    }
    // 展開済みノードの子に mid-turn 袋小路が含まれていないこと
    if (n.expanded) {
      for (const a of n.legal) {
        const c = n.children.get(a);
        if (c && !c.isTerminal && c.ctx.phase !== "TURN_START" && legalMicroActions(c.ctx).length === 0) {
          midTurnDeadEndsAsChildren++;
        }
        if (c) stack.push(c);
      }
    }
  }
  assert.equal(midTurnDeadEndsAsChildren, 0, "展開ノードの子に mid-turn 袋小路が残ってはならない");
});

test("★HIGH: 多段ターン境界(2回跨ぐ)でも符号が正しく伝播する", () => {
  // 人工4層鎖 A(white,ターン1)→B(black,ターン2)→C(white,ターン3)→D(white leaf)。
  // ターン境界を A→B, B→C の2回跨ぐ。全ノード N=0 の単一 simulate なら selectAction は
  // legal[0] を決定的に選ぶので、legal[0] に子をリンクして経路を固定する。
  // leaf(white) value>0 に対し mover 比較符号は A:+, B:-, C:+ になるはず（経路長非依存）。
  const mkState = (cp) => {
    const b = normalizeBoard(emptyBoard(), 4);
    b[0][0] = ["white"];
    return { board: b, summonCounts: { white: 4, black: 4 }, currentPlayer: cp, boardSize: 4 };
  };
  const A = makeNode(initTurnState(mkState("white")), "white", 4);
  const B = makeNode(initTurnState(mkState("black")), "black", 4);
  const C = makeNode(initTurnState(mkState("white")), "white", 4);
  const D = makeNode(initTurnState(mkState("white")), "white", 4);
  const aA = A.legal[0], aB = B.legal[0], aC = C.legal[0];
  const P1 = (idx) => { const p = new Float64Array(NUM_MICRO_ACTIONS); p[idx] = 1; return p; };
  // A,B,C を事前展開して legal[0] に子をリンク（D はリーフ=未展開）
  A.expanded = true; A.P = P1(aA); A.children.set(aA, B);
  B.expanded = true; B.P = P1(aB); B.children.set(aB, C);
  C.expanded = true; C.P = P1(aC); C.children.set(aC, D);

  const stub = { evaluatePolicy: () => ({ value: 5, policyLogits: new Float64Array(NUM_MICRO_ACTIONS) }) };
  // 単一 simulate: N=0 なので各ノードで legal[0] を選び A→B→C→D(expand)。leafMover=white(D)。
  const v = simulate(A, stub);
  assert.ok(v > 0, "leaf(white) value>0（value=5→v≈+0.98）");
  assert.ok(A.W[aA] > 0, "A(white) は leaf(white) と同符号 → 正");
  assert.ok(B.W[aB] < 0, "B(black) は leaf(white) と逆符号 → 負（境界1回目の反転）");
  assert.ok(C.W[aC] > 0, "C(white) は leaf(white) と同符号 → 正（境界2回跨いでも経路長非依存で正しい）");
  // 各エッジがちょうど1回ずつ訪問されたこと
  assert.equal(A.N[aA], 1);
  assert.equal(B.N[aB], 1);
  assert.equal(C.N[aC], 1);
});

test("終端ノード: 合法手ゼロは isTerminal かつ terminalValue=-1（mover視点）", () => {
  // black手番だが black は盤上0枚・召喚8済 → 合法手なし
  const board = emptyBoard();
  board[0][0] = ["white"];
  const node = makeRootNode({
    board,
    summonCounts: { white: 8, black: 8 },
    currentPlayer: "black",
    boardSize: 4,
  });
  assert.ok(node.isTerminal);
  assert.equal(node.terminalValue, -1);
  assert.equal(node.mover, "black");
});
