/**
 * mcts.mjs — AlphaZero移行 P3: PUCT モンテカルロ木探索。
 *
 * ノード = マイクロ状態（microActions.mjs の turnCtx）、エッジ = 81マイクロアクション。
 * 既存資産のみを read-only で利用する（js/ 以下は不変）:
 *   - 合法手/遷移: microActions.mjs（legalMicroActions / applyMicroAction / initTurnState）
 *   - ネット評価: network.js evaluatePolicy(indices, scalars, turnContext)
 *   - 入力生成: features.js extractFeatures + azFeatures.mjs extractTurnContext
 *
 * ★符号規約（最重要・PLAN §5.1 / §10）★
 *   - value は「そのマイクロ状態の手番プレイヤー(mover)視点の勝率」を [-1,+1] に写像
 *     したもの（v = 2*sigmoid(rawValue) - 1）。終端（合法手なし=moverの負け）は -1。
 *   - ターン内マイクロ遷移では手番が変わらない。END_TURN・召喚・排除の「done」でのみ
 *     手番が交代する。各ノードは自分の mover を保持する。
 *   - backup では leaf の value（leaf.mover 視点）を各祖先ノード n に反映する際、
 *     符号を sign = (n.mover === leaf.mover) ? +1 : -1 で一括変換する
 *     （マイクロ遷移かターン交代かを毎回判定するより mover 比較の方が堅牢）。
 *   - Q(s,a) は各ノードの mover 視点で W/N として保持し、argmax する。
 */

import {
  initTurnState,
  legalMicroActions,
  applyMicroAction,
  NUM_MICRO_ACTIONS,
} from "./microActions.mjs";
import { extractFeatures } from "../nnue/features.js";
import { extractTurnContext } from "./azFeatures.mjs";

export const DEFAULT_C_PUCT = 1.5;

function opponentOf(player) {
  return player === "white" ? "black" : "white";
}

/**
 * このマイクロ状態からターンを完了（done）できる継続が存在するか。
 * microActions.mjs の契約: END は effective-pass 以外で常に合法なので、END が合法な
 * 通常のマイクロ状態では即 true（短絡）。effective-pass の袋小路のみ深く探索するが、
 * ctx.historyKeys（反復禁止）と hop 上限により継続は有限で必ず停止する。
 * @param {object} ctx - turnCtx
 * @returns {boolean}
 */
export function canComplete(ctx) {
  const legal = legalMicroActions(ctx);
  for (const a of legal) {
    const next = applyMicroAction(ctx, a);
    if (next.done) return true; // END/召喚/排除でターン完了 → 完了可能
    if (canComplete(next)) return true;
  }
  return false;
}

/**
 * ctx の合法手のうち「実際に選ぶ価値のある」手（＝袋小路に落ちない手）を返す。
 * done で終わる手は常に採用可（相手 TURN_START＝勝ち/継続の分岐）。done でない手は、
 * 遷移先からターンを完了できる場合のみ採用する（microActions.mjs 契約:
 * AUGMENT/CHAIN の袋小路＝実質パス盤面＋反復禁止で全刈られた行き止まりは「敗北ではなく放棄」）。
 * @param {object} ctx - turnCtx
 * @returns {number[]}
 */
export function viableActions(ctx) {
  return legalMicroActions(ctx).filter((a) => {
    const next = applyMicroAction(ctx, a);
    return next.done || canComplete(next);
  });
}

/**
 * マイクロ状態からノードを生成する。
 *
 * ★isTerminal の定義（CRITICAL: microActions.mjs 195-211 の契約に厳密準拠）★
 *   TURN_START で合法手ゼロ = 真の敗北（hasAnyLegalTurn===false と同値）→ terminal, -1。
 *   AUGMENT/CHAIN で合法手ゼロ = 実質パス盤面＋反復禁止で全刈られた袋小路。
 *     これは「敗北ではなく放棄すべき行き止まり」であり terminal ではない（虚偽の -1 を
 *     全木に符号反転伝播させない）。袋小路への遷移は viableActions が親側で除外する。
 *
 * @param {object} ctx - turnCtx
 * @param {string} mover - このノードの手番プレイヤー
 * @param {number} boardSize
 */
export function makeNode(ctx, mover, boardSize) {
  const legal = legalMicroActions(ctx);
  // 真の終端は「TURN_START で合法ゼロ」＝手番プレイヤーに合法ターンが1つもない場合のみ。
  const isTerminal = ctx.phase === "TURN_START" && legal.length === 0;
  return {
    ctx,
    mover,
    boardSize,
    isTerminal,
    terminalValue: isTerminal ? -1 : 0, // mover 視点（真の敗北 = -1）
    expanded: false,
    legal, // expandNode で viableActions に絞り込む
    P: null, // Float64Array(81)（viable マスク後 softmax）
    N: new Float64Array(NUM_MICRO_ACTIONS),
    W: new Float64Array(NUM_MICRO_ACTIONS),
    Nnode: 0, // ノード全体の訪問数（= Σ_a N[a]）
    children: new Map(),
  };
}

/** ゲーム状態（board, summonCounts, currentPlayer, boardSize）からルートノードを作る。 */
export function makeRootNode(state) {
  const ctx = initTurnState(state);
  return makeNode(ctx, state.currentPlayer, state.boardSize);
}

/**
 * ノードの子（action 採用後のマイクロ状態）を取得または生成する。
 * done（ターン終了）なら相手手番の新しい TURN_START ノードになる。
 */
export function childFor(node, action) {
  const existing = node.children.get(action);
  if (existing) return existing;

  const next = applyMicroAction(node.ctx, action);
  let child;
  if (next.done) {
    const opp = opponentOf(node.mover);
    const oppCtx = initTurnState({
      board: next.board,
      summonCounts: next.summonCounts,
      currentPlayer: opp,
      boardSize: node.boardSize,
    });
    child = makeNode(oppCtx, opp, node.boardSize);
  } else {
    child = makeNode(next, node.mover, node.boardSize);
  }
  node.children.set(action, child);
  return child;
}

/**
 * リーフノードを展開する: ネット評価で P（合法手マスク後 softmax）と value を求める。
 * @returns {number} value（このノードの mover 視点, [-1,+1]）
 */
export function expandNode(node, net) {
  const microState = {
    board: node.ctx.board,
    summonCounts: node.ctx.summonCounts,
    currentPlayer: node.mover,
    boardSize: node.boardSize,
  };
  // ★展開時に袋小路へ落ちる手を除外（viableActions）。以降 node.legal は viable のみ。
  //   これにより mid-turn 袋小路が子として生成されず、虚偽の -1 が混入しない。
  node.legal = viableActions(node.ctx);

  const { indices, scalars } = extractFeatures(microState, node.mover);
  const turnContext = Array.from(extractTurnContext(node.ctx));
  const { value, policyLogits } = net.evaluatePolicy(indices, scalars, turnContext);

  // 合法手のみ softmax
  let maxLogit = -Infinity;
  for (const a of node.legal) if (policyLogits[a] > maxLogit) maxLogit = policyLogits[a];
  let sum = 0;
  const exps = new Float64Array(NUM_MICRO_ACTIONS);
  for (const a of node.legal) {
    const e = Math.exp(policyLogits[a] - maxLogit);
    exps[a] = e;
    sum += e;
  }
  const P = new Float64Array(NUM_MICRO_ACTIONS);
  for (const a of node.legal) P[a] = exps[a] / sum;
  node.P = P;
  node.expanded = true;

  const winProb = 1 / (1 + Math.exp(-value)); // mover 視点勝率 [0,1]
  return 2 * winProb - 1; // [-1,+1]
}

/**
 * PUCT 選択: a* = argmax_a [ Q(s,a) + c_puct·P(s,a)·√ΣN / (1+N(s,a)) ]。
 * 未訪問エッジの Q は 0（中立）。同点は legal 配列内の先着（＝低 index 寄り）で決定的に破る。
 */
export function selectAction(node, cPuct = DEFAULT_C_PUCT) {
  const sqrtTotal = Math.sqrt(node.Nnode);
  let bestScore = -Infinity;
  let bestAction = node.legal[0];
  for (const a of node.legal) {
    const q = node.N[a] > 0 ? node.W[a] / node.N[a] : 0;
    const u = cPuct * node.P[a] * sqrtTotal / (1 + node.N[a]);
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      bestAction = a;
    }
  }
  return bestAction;
}

/**
 * 1シミュレーション: 選択 → 展開/終端評価 → backup。
 * @returns {number} この simulation で得た leaf value（leaf.mover 視点）
 */
export function simulate(root, net, cPuct = DEFAULT_C_PUCT) {
  const path = [];
  let node = root;
  while (node.expanded && !node.isTerminal) {
    const a = selectAction(node, cPuct);
    path.push({ node, action: a });
    node = childFor(node, a);
  }

  let value;
  if (node.isTerminal) {
    value = node.terminalValue; // -1（node.mover の負け）
  } else {
    value = expandNode(node, net); // node.mover 視点
  }
  const leafMover = node.mover;

  // ★符号変換つき backup（mover 比較で一括）★
  for (const { node: n, action } of path) {
    const sign = n.mover === leafMover ? 1 : -1;
    n.N[action] += 1;
    n.Nnode += 1;
    n.W[action] += sign * value;
  }
  return value;
}

/**
 * MCTS を回す。simulations（固定回数）または deadline（時刻ミリ秒）で停止。
 * deadline 指定時も最低 minSims 回は回す（末端マイクロ手で 0 回を避ける）。
 * @returns {number} 実行した simulation 回数
 */
export function runMCTS(root, net, opts = {}) {
  const {
    cPuct = DEFAULT_C_PUCT,
    simulations = null,
    deadline = null,
    minSims = 8,
  } = opts;
  if (simulations == null && deadline == null) {
    throw new Error("runMCTS: simulations または deadline のいずれかが必須");
  }
  let sims = 0;
  for (;;) {
    if (simulations != null && sims >= simulations) break;
    if (deadline != null && sims >= minSims && Date.now() >= deadline) break;
    simulate(root, net, cPuct);
    sims++;
    // 固定回数と deadline の両方が無い経路は上でthrow済み。deadline のみで minSims 未満は継続。
    if (simulations == null && deadline != null && sims >= minSims && Date.now() >= deadline) break;
  }
  return sims;
}

/**
 * ルートで訪問数最大のアクションを返す（決定的: 同点は低 index）。
 * temperature>0 の場合は訪問数を温度付き確率でサンプリング（自己対戦 P4 用、既定は決定的）。
 */
export function bestActionByVisits(root, { temperature = 0, rng = Math.random } = {}) {
  if (temperature > 0) {
    const weights = [];
    let sum = 0;
    for (const a of root.legal) {
      const w = Math.pow(root.N[a], 1 / temperature);
      weights.push([a, w]);
      sum += w;
    }
    if (sum > 0) {
      let x = rng() * sum;
      for (const [a, w] of weights) {
        x -= w;
        if (x <= 0) return a;
      }
      return weights[weights.length - 1][0];
    }
  }
  let bestN = -Infinity;
  let bestA = root.legal[0];
  for (const a of root.legal) {
    if (root.N[a] > bestN) {
      bestN = root.N[a];
      bestA = a;
    }
  }
  return bestA;
}

/** ルートの Q(s,a)（mover 視点）を返すヘルパ（テスト・解析用）。未訪問は null。 */
export function rootQ(root, action) {
  return root.N[action] > 0 ? root.W[action] / root.N[action] : null;
}
