/**
 * analysis.js — 局面解析コアモジュール（Phase 1: マルチPV解析）
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール。
 * 指定局面から合法な全ターンを列挙し、NNUE 評価関数を注入した
 * findBestTurn（search.js）でそれぞれの手を読み切り、
 * 上位 topN 手を候補（candidates）としてスコア降順に返す。
 *
 * 本モジュールは純関数のみで構成され、DOM・Worker 依存を持たない。
 */

import { normalizeBoard, maxSummonsFor, checkWinCondition } from "../gameLogic.js";
import { generateTurns } from "../ai/moveGen.js";
import { findBestTurn } from "../ai/search.js";
import { WIN_SCORE } from "../ai/evaluate.js";
import { diffToNotation } from "../coords.js";

// ───────────────────────── ユーティリティ ─────────────────────────

/** プレイヤーを反転する */
function opponent(player) {
  return player === "white" ? "black" : "white";
}

/**
 * シグモイド関数。生スコア（logit）を [0,1] の勝率に変換する。
 * @param {number} x
 * @returns {number}
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// ───────────────────────── 主要エクスポート ─────────────────────────

/**
 * 指定局面の上位候補手を NNUE 評価付き探索で解析する。
 *
 * @param {{ board: Array[][]|Array, summonCounts: {white:number,black:number},
 *            currentPlayer: string, boardSize: number }} state
 *   board は正規化済み・denormalized のどちらでも可（内部で normalizeBoard する）。
 * @param {{
 *   evaluateState: (state:object, povPlayer?:string) => number,
 *   meta?: object
 * }} net - network.js の createNetwork/loadNetwork が返す network オブジェクト。
 * @param {{
 *   maxDepth?: number,
 *   timeBudgetMs?: number,
 *   topN?: number,
 *   rng?: () => number,
 *   forwardPruneWidth?: number
 * }} [options]
 * @returns {{
 *   current: { score: number, winProb: number, whiteWinProb: number },
 *   totalCandidates: number,
 *   candidates: Array<{
 *     actions: object[],
 *     board: Array[][],
 *     summonCounts: {white:number,black:number},
 *     notation: string,
 *     score: number,
 *     winProb: number,
 *     isWin: boolean
 *   }>
 * }}
 *   totalCandidates は topN で切る前の合法手総数（呼び出し側が時間予算を
 *   分岐数で分割するのに使う）。
 */
export function analyzeTopMoves(state, net, options = {}) {
  const {
    maxDepth = 8,
    timeBudgetMs = 1000,
    topN = 3,
    rng = Math.random,
    forwardPruneWidth,
  } = options;

  const { boardSize, currentPlayer, summonCounts } = state;
  const normalized = normalizeBoard(state.board, boardSize);
  const nstate = { board: normalized, summonCounts, currentPlayer, boardSize };

  const meta = { maxSummons: maxSummonsFor(boardSize), boardSize };
  const opp = opponent(currentPlayer);

  const cands = generateTurns(nstate);
  if (cands.length === 0) {
    return {
      current: { score: 0, winProb: sigmoid(0), whiteWinProb: sigmoid(0) },
      totalCandidates: 0,
      candidates: [],
    };
  }

  const evalFn = (s, p) => net.evaluateState(s, p);

  const scored = cands.map((c) => {
    const childState = {
      board: c.board,
      summonCounts: c.summonCounts,
      currentPlayer: opp,
      boardSize,
    };

    let score;
    let isWin;
    if (checkWinCondition(childState, meta)) {
      score = WIN_SCORE;
      isWin = true;
    } else {
      const searchOptions = { maxDepth, timeBudgetMs, rng, evalFn };
      if (forwardPruneWidth !== undefined) {
        searchOptions.forwardPruneWidth = forwardPruneWidth;
      }
      const result = findBestTurn(childState, searchOptions);
      score = -result.score;
      isWin = false;
    }

    let notation = diffToNotation(normalized, c.board, boardSize);
    if (isWin) notation += "#";

    return {
      actions: c.actions,
      board: c.board,
      summonCounts: c.summonCounts,
      notation,
      score,
      winProb: sigmoid(score),
      isWin,
    };
  });

  // score 降順にソート（安定ソート: 同点は生成順を維持）
  scored.sort((a, b) => b.score - a.score);

  const candidates = scored.slice(0, topN);

  const currentScore = candidates.length ? candidates[0].score : 0;
  const winProb = sigmoid(currentScore);
  const whiteScore = currentPlayer === "white" ? currentScore : -currentScore;
  const whiteWinProb = sigmoid(whiteScore);

  return {
    current: { score: currentScore, winProb, whiteWinProb },
    totalCandidates: cands.length,
    candidates,
  };
}
