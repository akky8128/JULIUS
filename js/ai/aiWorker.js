/**
 * aiWorker.js — CPUプレイヤー用 Web Worker (ESモジュール)
 *
 * メインスレッドから以下のメッセージを受信する:
 *   { requestId: number, state: { board, summonCounts, currentPlayer, boardSize }, level: 1|2|3 }
 *
 * 以下のメッセージを返す（成功時）:
 *   { requestId, turn: { actions, board, summonCounts } | null, info: { depthReached?, nodes?, elapsedMs } }
 *
 * エラー時:
 *   { requestId, error: string }
 */

import { normalizeBoard, maxSummonsFor, checkWinCondition } from "../gameLogic.js";
import { generateTurns } from "./moveGen.js";
import { evaluate, WEIGHTS } from "./evaluate.js";
import { findBestTurn } from "./search.js";

// ───────────────────────── ユーティリティ ─────────────────────────

/** プレイヤーを反転する */
function opponent(player) {
  return player === "white" ? "black" : "white";
}

// ───────────────────────── レベル別 chooseTurn ─────────────────────────

/**
 * レベル1: ランダムプレイヤー（即勝ちがあればそれを選ぶ）
 */
function level1Choose(state) {
  const { currentPlayer, boardSize } = state;
  const maxSummons = maxSummonsFor(boardSize);
  const meta = { maxSummons, boardSize };
  const opp = opponent(currentPlayer);

  const turns = generateTurns(state);
  if (turns.length === 0) return { turn: null, info: { elapsedMs: 0 } };

  // 即勝ちがあれば選ぶ
  for (const candidate of turns) {
    const nextState = {
      board: candidate.board,
      summonCounts: candidate.summonCounts,
      currentPlayer: opp,
      boardSize,
    };
    if (checkWinCondition(nextState, meta)) {
      return { turn: candidate, info: { elapsedMs: 0 } };
    }
  }

  // ランダムに選ぶ
  const chosen = turns[Math.floor(Math.random() * turns.length)];
  return { turn: chosen, info: { elapsedMs: 0 } };
}

/**
 * レベル2: 貪欲プレイヤー（即勝ち優先、次に evaluate スコア最大、同点ランダム）
 */
function level2Choose(state) {
  const t0 = Date.now();
  const { currentPlayer, boardSize } = state;
  const maxSummons = maxSummonsFor(boardSize);
  const meta = { maxSummons, boardSize };
  const opp = opponent(currentPlayer);

  const turns = generateTurns(state);
  if (turns.length === 0) return { turn: null, info: { elapsedMs: Date.now() - t0 } };

  // 即勝ちチェック
  for (const candidate of turns) {
    const nextState = {
      board: candidate.board,
      summonCounts: candidate.summonCounts,
      currentPlayer: opp,
      boardSize,
    };
    if (checkWinCondition(nextState, meta)) {
      return { turn: candidate, info: { elapsedMs: Date.now() - t0 } };
    }
  }

  // evaluate スコアで最大候補を選ぶ（同点はランダム）
  let bestScore = -Infinity;
  let bestCandidates = [];

  for (const candidate of turns) {
    const resultState = {
      board: candidate.board,
      summonCounts: candidate.summonCounts,
      currentPlayer: opp,
      boardSize,
    };
    const score = evaluate(resultState, currentPlayer, WEIGHTS);
    if (score > bestScore) {
      bestScore = score;
      bestCandidates = [candidate];
    } else if (score === bestScore) {
      bestCandidates.push(candidate);
    }
  }

  const chosen = bestCandidates[Math.floor(Math.random() * bestCandidates.length)];
  return { turn: chosen, info: { elapsedMs: Date.now() - t0 } };
}

/**
 * レベル3: 反復深化 α-β 探索（depth 4, 1000ms）
 */
function level3Choose(state) {
  const result = findBestTurn(state, {
    maxDepth: 4,
    timeBudgetMs: 1000,
    weights: WEIGHTS,
    rng: Math.random,
  });
  return {
    turn: result.turn,
    info: {
      depthReached: result.depthReached,
      nodes: result.nodes,
      elapsedMs: result.elapsedMs,
    },
  };
}

// ───────────────────────── メッセージハンドラ ─────────────────────────

self.onmessage = (event) => {
  const { requestId, state: rawState, level } = event.data;

  try {
    // board を正規化（denormalized (0 や null) でも normalized でも受け付ける）
    const boardSize = rawState.boardSize;
    const normalizedBoard = normalizeBoard(rawState.board, boardSize);

    const state = {
      board: normalizedBoard,
      summonCounts: rawState.summonCounts,
      currentPlayer: rawState.currentPlayer,
      boardSize,
    };

    let result;
    if (level === 1) {
      result = level1Choose(state);
    } else if (level === 3) {
      result = level3Choose(state);
    } else {
      // デフォルトはレベル2（ふつう）
      result = level2Choose(state);
    }

    self.postMessage({
      requestId,
      turn: result.turn,
      info: result.info,
    });
  } catch (err) {
    self.postMessage({
      requestId,
      error: err.message || String(err),
    });
  }
};
