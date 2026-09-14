/**
 * mctsPlayer.mjs — AlphaZero移行 P3: MCTS を「1ターンを指すエージェント」にラップ。
 *
 * 時間予算はターン単位（1ply あたり）で受け取り、そのターンの全マイクロ決定に配分する。
 * これにより gen137+αβ（findBestTurn に timeBudgetMs/ply を渡す）と公平に比較できる。
 *
 * 返り値は generateTurns 候補と同形式 {actions?, board, summonCounts, microActions} を返す
 * （board/summonCounts は最終盤面。actions は移動系のみ再構成が難しいため microActions を併記）。
 */

import {
  initTurnState,
  applyMicroAction,
} from "../../core/az/microActions.mjs";
import { makeNode, runMCTS, bestActionByVisits, viableActions } from "../../core/az/mcts.mjs";
import { DEFAULT_C_PUCT } from "../../core/az/mcts.mjs";

/**
 * 1ターンを MCTS で指す。
 *
 * @param {object} state - {board(正規化済み), summonCounts, currentPlayer, boardSize}
 * @param {object} net - createNetwork 由来（evaluatePolicy を持つ）
 * @param {object} opts
 *   - timeBudgetMs: ターン全体の時間予算（既定 120）。simulations 指定時は無視。
 *   - simulations: マイクロ手あたり固定シミュレーション数（指定時は時間予算より優先）
 *   - cPuct, minSims, temperature, rng, maxMicroSteps
 * @returns {{ board, summonCounts, microActions:number[], nodesInfo:object }}
 */
export function playTurnMcts(state, net, opts = {}) {
  const {
    timeBudgetMs = 120,
    simulations = null,
    cPuct = DEFAULT_C_PUCT,
    minSims = 8,
    temperature = 0,
    rng = Math.random,
    maxMicroSteps = 64,
  } = opts;

  const boardSize = state.boardSize;
  const mover = state.currentPlayer;
  let ctx = initTurnState(state);
  const microActions = [];
  let totalSims = 0;

  // 時間予算モード: ターン全体で共有する deadline。各マイクロ決定はこの deadline まで
  // 回すが、後続のマイクロ手にも時間が残るよう「経過に応じて自然に短くなる」。
  // マイクロ手数は少数（実測 平均~2）なので deadline 共有で総 ms ≈ timeBudgetMs に収まる。
  const turnDeadline = simulations == null ? Date.now() + timeBudgetMs : null;

  for (let step = 0; step < maxMicroSteps; step++) {
    // viableActions: 袋小路（AUGMENT/CHAIN で完了不能な行き止まり）へ落ちる手を除外。
    // mcts.mjs の展開時フィルタと同一意味論に揃える。
    const legal = viableActions(ctx);
    if (legal.length === 0) {
      // viable が空 = このマイクロ状態からターンを完了できない。bestActionByVisits は
      // viable のみ選ぶため実プレイで新規ノードがここに来ることはないが、防御的に検知。
      throw new Error(`mctsPlayer: no viable micro-action at step ${step} (phase=${ctx.phase})`);
    }
    if (legal.length === 1) {
      // 選択の余地なし。探索せず即適用（時間節約）。
      ctx = applyMicroAction(ctx, legal[0]);
      microActions.push(legal[0]);
      if (ctx.done) break;
      continue;
    }

    const root = makeNode(ctx, mover, boardSize);
    const sims = runMCTS(root, net, {
      cPuct,
      simulations,
      deadline: turnDeadline,
      minSims,
    });
    totalSims += sims;

    const action = bestActionByVisits(root, { temperature, rng });
    ctx = applyMicroAction(ctx, action);
    microActions.push(action);
    if (ctx.done) break;
  }

  if (!ctx.done) {
    throw new Error("mctsPlayer: turn did not terminate within maxMicroSteps");
  }

  return {
    board: ctx.board,
    summonCounts: ctx.summonCounts,
    microActions,
    nodesInfo: { totalSims, microSteps: microActions.length },
  };
}

/**
 * mctsPlayer をエージェントインターフェース（selfplay/対局ハーネス用）にする。
 * chooseTurn(state) → {board, summonCounts, microActions} を返す。
 */
export function createMctsPlayer(net, opts = {}) {
  return {
    name: opts.name || "mcts",
    chooseTurn(state) {
      return playTurnMcts(state, net, opts);
    },
  };
}
