/**
 * players.js — AI プレイヤーファクトリ
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール。
 */

import { generateTurns } from "./moveGen.js";
import { checkWinCondition, maxSummonsFor } from "../gameLogic.js";
import { evaluate, WEIGHTS } from "./evaluate.js";
import { findBestTurn } from "./search.js";

// ───────────────────────── 内部ユーティリティ ─────────────────────────

function opponent(player) {
  return player === "white" ? "black" : "white";
}

// ───────────────────────── プレイヤーファクトリ ─────────────────────────

/**
 * ランダムプレイヤーを生成する。
 * @param {() => number} rng - [0,1) の乱数を返す関数
 * @returns {{ name: string, chooseTurn(state): object|null }}
 */
export function randomPlayer(rng) {
  return {
    name: "random",
    chooseTurn(state) {
      const turns = generateTurns(state);
      if (turns.length === 0) return null;
      return turns[Math.floor(rng() * turns.length)];
    },
  };
}

/**
 * 貪欲プレイヤーを生成する。
 * 即勝ち候補があれば即座に選択し、
 * なければ evaluate() でスコアが最大の候補を選ぶ。
 * 同点はランダムに破る（決定論的な tie-breaking は move loop の原因になる）。
 *
 * @param {() => number} rng - [0,1) の乱数を返す関数
 * @param {typeof WEIGHTS} weights
 * @returns {{ name: string, chooseTurn(state): object|null }}
 */
export function greedyPlayer(rng, weights = WEIGHTS) {
  return {
    name: "greedy",
    chooseTurn(state) {
      const { currentPlayer, boardSize } = state;
      const turns = generateTurns(state);
      if (turns.length === 0) return null;

      const maxSummons = maxSummonsFor(boardSize);
      const opp = opponent(currentPlayer);
      const meta = { maxSummons, boardSize };

      // 即勝ちチェック: 相手の手番になった局面で相手に合法手がなければ勝ち
      for (const candidate of turns) {
        const nextState = {
          board: candidate.board,
          summonCounts: candidate.summonCounts,
          currentPlayer: opp,  // 次のターンは相手
          boardSize,
        };
        if (checkWinCondition(nextState, meta)) {
          return candidate;
        }
      }

      // 最大スコアの候補を選択（同点はランダム破り）
      let bestScore = -Infinity;
      let bestCandidates = [];

      for (const candidate of turns) {
        const resultState = {
          board: candidate.board,
          summonCounts: candidate.summonCounts,
          currentPlayer: opp,  // 次のターンは相手（evaluate は player 視点で計算）
          boardSize,
        };
        const score = evaluate(resultState, currentPlayer, weights);

        if (score > bestScore) {
          bestScore = score;
          bestCandidates = [candidate];
        } else if (score === bestScore) {
          bestCandidates.push(candidate);
        }
      }

      // 同点候補からランダムに1つ選ぶ
      return bestCandidates[Math.floor(rng() * bestCandidates.length)];
    },
  };
}

/**
 * 反復深化 α-β 探索プレイヤーを生成する。
 *
 * @param {() => number} rng - [0,1) の乱数を返す関数（タイブレーク用）
 * @param {{
 *   maxDepth?: number,
 *   timeBudgetMs?: number,
 *   weights?: typeof WEIGHTS
 * }} options
 * @returns {{ name: string, chooseTurn(state): object|null,
 *             lastSearchInfo: {depthReached:number, nodes:number, elapsedMs:number}|null }}
 */
export function searchPlayer(rng, { maxDepth = 4, timeBudgetMs = 1000, weights } = {}) {
  // 最後の探索情報（selfplay.mjs の統計収集用）
  let lastSearchInfo = null;

  return {
    name: `search(d${maxDepth},${timeBudgetMs}ms)`,
    get lastSearchInfo() { return lastSearchInfo; },
    chooseTurn(state) {
      const result = findBestTurn(state, { maxDepth, timeBudgetMs, weights, rng });
      lastSearchInfo = {
        depthReached: result.depthReached,
        nodes:        result.nodes,
        elapsedMs:    result.elapsedMs,
      };
      return result.turn;
    },
  };
}
