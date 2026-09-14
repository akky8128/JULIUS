/**
 * turnToMicro.mjs — AlphaZero移行 P2: generateTurns 候補のマイクロアクション列分解。
 *
 * generateTurns（core/ai/moveGen.js）の候補が持つ actions 配列
 * （summon / eliminate / move{from,to,count} のブロック移動列）を、
 * microActions.mjs の81ノード規約のマイクロアクション列に分解する純関数。
 *
 * 対応規則（PLAN.md §3.2）:
 *   - summon/eliminate → 該当ノード1つ（採用即ターン終了、ENDノード不要）
 *   - 初手 move(count=k) → TURN_START での移動1回 + AUGMENT 再採用 (k-1)回
 *   - チェーンの move(count=k) → チェーン遷移1回 + CHAIN 再採用 (k-1)回
 *   - 移動列の末尾に END_TURN_ACTION
 */

import { hashPosition } from "../../core/ai/moveGen.js";
import {
  initTurnState,
  legalMicroActions,
  applyMicroAction,
  moveNode,
  summonNode,
  END_TURN_ACTION,
} from "../../core/az/microActions.mjs";

/** (from,to) の1マス差分から DIRS 規約の方向インデックス(0=上,1=下,2=左,3=右)を求める。 */
function dirOf(from, to) {
  const dr = to.r - from.r;
  const dc = to.c - from.c;
  if (dr === -1 && dc === 0) return 0;
  if (dr === 1 && dc === 0) return 1;
  if (dr === 0 && dc === -1) return 2;
  if (dr === 0 && dc === 1) return 3;
  throw new Error(`turnToMicro: non-adjacent move from (${from.r},${from.c}) to (${to.r},${to.c})`);
}

/**
 * candidate.actions からマイクロアクションindex列を組み立てる（検証なしの純変換）。
 * @returns {number[]}
 */
export function turnActionsToMicroIndices(actions, boardSize) {
  const indices = [];
  let anyMove = false;
  for (const action of actions) {
    if (action.type === "summon" || action.type === "eliminate") {
      if (actions.length !== 1) {
        throw new Error(`turnToMicro: ${action.type} must be the only action in a turn`);
      }
      return [summonNode(action.r, action.c, boardSize)];
    }
    if (action.type !== "move") throw new Error(`turnToMicro: unknown action type ${action.type}`);
    anyMove = true;
    const dir = dirOf(action.from, action.to);
    const node = moveNode(action.from.r, action.from.c, dir, boardSize);
    for (let k = 0; k < action.count; k++) indices.push(node);
  }
  if (anyMove) indices.push(END_TURN_ACTION);
  return indices;
}

/**
 * generateTurns の候補1つをマイクロアクション列に分解し、文法上の正当性を検証する。
 *
 * 各ステップで legalMicroActions(ctx) に含まれることをアサートしながら
 * applyMicroAction で進め、最終盤面が candidate.board / candidate.summonCounts と
 * hashPosition 一致することを検証する。違反があれば Error を投げる（握りつぶさない）。
 *
 * @param {{board:Array, summonCounts:object, currentPlayer:string, boardSize:number}} state
 * @param {{actions:object[], board:Array, summonCounts:object}} candidate
 * @returns {{ actionIndices: number[], contexts: object[] }}
 *   contexts[i] は actionIndices[i] を採用する「直前」の turnCtx
 *   （azFeatures.extractTurnContext の入力に使う。policy 教師 = そのctxでの選択肢）。
 */
export function decomposeTurn(state, candidate) {
  const boardSize = state.boardSize;
  const actionIndices = turnActionsToMicroIndices(candidate.actions, boardSize);
  const contexts = [];

  let ctx = initTurnState(state);
  for (const actionIdx of actionIndices) {
    const legal = legalMicroActions(ctx);
    if (!legal.includes(actionIdx)) {
      throw new Error(
        `turnToMicro: action ${actionIdx} not legal at phase=${ctx.phase} ` +
          `(legal=[${legal.join(",")}])\nactions=${JSON.stringify(candidate.actions)}\n` +
          `state=${JSON.stringify({ board: state.board, summonCounts: state.summonCounts, currentPlayer: state.currentPlayer })}`
      );
    }
    contexts.push(ctx);
    ctx = applyMicroAction(ctx, actionIdx);
  }

  if (!ctx.done) {
    throw new Error(`turnToMicro: sequence did not terminate the turn (phase=${ctx.phase})`);
  }

  const finalHash = hashPosition(ctx.board, ctx.summonCounts);
  const expectedHash = hashPosition(candidate.board, candidate.summonCounts);
  if (finalHash !== expectedHash) {
    throw new Error(
      `turnToMicro: final board mismatch\nexpected=${expectedHash}\nactual  =${finalHash}\n` +
        `actions=${JSON.stringify(candidate.actions)}`
    );
  }

  return { actionIndices, contexts };
}
