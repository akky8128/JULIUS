/**
 * nnueWorker.js — Step 7「NNUE Lab」用 CPU プレイヤー Web Worker (ESモジュール)
 *
 * julius/public/js/workers/aiWorker.js（本番用・変更禁止）と同じメッセージ仕様を踏襲しつつ、
 * 将来 NNUE 評価関数を差し込むための拡張ポイントを用意した開発専用 Worker。
 * 本番の aiWorker.js は一切変更していない。
 *
 * メインスレッドから以下のメッセージを受信する:
 *   {
 *     requestId: number,
 *     state: { board, summonCounts, currentPlayer, boardSize },
 *     level: 1 | 2 | 3,
 *     options?: {
 *       maxDepth?: number,        // level3/NNUE 探索の最大深さ（既定4）
 *       timeBudgetMs?: number,    // level3/NNUE 探索の思考時間予算（既定1000ms）
 *       net?: unknown,            // TODO(Step5): 将来のNNUEモデル指定（未実装）
 *     },
 *   }
 *
 * 以下のメッセージを返す（成功時）:
 *   {
 *     requestId,
 *     turn: { actions, board, summonCounts } | null,
 *     info: { depthReached?, nodes?, elapsedMs, score? },
 *   }
 *
 * エラー時:
 *   { requestId, error: string }
 */

import { normalizeBoard, maxSummonsFor, checkWinCondition } from "../../../core/gameLogic.js";
import { generateTurns } from "../../../core/ai/moveGen.js";
import { evaluate, WEIGHTS } from "../../../core/ai/evaluate.js";
import { findBestTurn } from "../../../core/ai/search.js";
import { createNetworkFromBuffer } from "../../../core/nnue/network.js";

// ───────────────────────── ユーティリティ ─────────────────────────

/** プレイヤーを反転する */
function opponent(player) {
  return player === "white" ? "black" : "white";
}

// ───────────────────────── NNUE ネットワークのロード（Worker用） ─────────────────────────
//
// network.js の loadNetwork() は `window` の有無でブラウザ/Node を判定するが、
// Web Worker には window が存在しないため誤って Node 経路に落ちる。ここでは
// fetch で .json / .bin を取得し createNetworkFromBuffer で構築する専用経路を持つ。
// path ごとに Promise をキャッシュし、同一モデルの再ロードを避ける。

/** @type {Map<string, Promise<ReturnType<typeof createNetworkFromBuffer>>>} */
const netCache = new Map();

/**
 * netPath（.json）から NNUE network をロードする（キャッシュ付き）。
 * @param {string} jsonPath - 例 "research/models/gen012.json"（.bin は自動導出）
 * @returns {Promise<ReturnType<typeof createNetworkFromBuffer>>}
 */
function loadNetForWorker(jsonPath) {
  const normalized = jsonPath.replace(/\.bin$/, ".json");
  let cached = netCache.get(normalized);
  if (cached) return cached;

  // Worker 内の相対 fetch は Worker スクリプト(core/ai/)基準で解決されてしまうため、
  // ページ origin を基準に絶対URL化する（"/research/models/..." でも "research/models/..." でも動く）。
  const jsonUrl = new URL(normalized, self.location.origin).href;
  const binPath = jsonUrl.replace(/\.json$/, ".bin");
  cached = Promise.all([
    fetch(jsonUrl).then((r) => {
      if (!r.ok) throw new Error(`モデルJSONの取得に失敗: ${normalized} (${r.status})`);
      return r.json();
    }),
    fetch(binPath).then((r) => {
      if (!r.ok) throw new Error(`モデル重みの取得に失敗: ${binPath} (${r.status})`);
      return r.arrayBuffer();
    }),
  ]).then(([meta, buf]) => createNetworkFromBuffer(buf, meta));

  netCache.set(normalized, cached);
  return cached;
}

// level1/2 の静的評価は常に従来の heuristic を使う（NNUE は探索を伴う level 対象）。
function resolveEvaluator(_options) {
  return { evaluateFn: evaluate, weights: WEIGHTS };
}

/**
 * NNUE 評価関数を使う反復深化 α-β 探索。
 * options.net（.json パス）で指定したモデルをロードし、findBestTurn の
 * evalFn に net.evaluateState を注入する。
 *
 * @param {object} state
 * @param {{ maxDepth?: number, timeBudgetMs?: number, net: string }} options
 * @returns {Promise<{ turn: object|null, info: object }>}
 */
async function nnueChoose(state, options) {
  const maxDepth = (options && Number.isFinite(options.maxDepth)) ? options.maxDepth : 64;
  const timeBudgetMs = (options && Number.isFinite(options.timeBudgetMs)) ? options.timeBudgetMs : 1000;

  const net = await loadNetForWorker(options.net);

  const result = findBestTurn(state, {
    maxDepth,
    timeBudgetMs,
    rng: Math.random,
    evalFn: (s, p) => net.evaluateState(s, p),
  });

  return {
    turn: result.turn,
    info: {
      depthReached: result.depthReached,
      nodes: result.nodes,
      elapsedMs: result.elapsedMs,
      score: result.score,
    },
  };
}

// ───────────────────────── レベル別 chooseTurn ─────────────────────────

/**
 * レベル1: ランダムプレイヤー（即勝ちがあればそれを選ぶ）
 * 選んだ手の評価値も可能なら info.score に含める。
 */
function level1Choose(state, options) {
  const t0 = Date.now();
  const { currentPlayer, boardSize } = state;
  const maxSummons = maxSummonsFor(boardSize);
  const meta = { maxSummons, boardSize };
  const opp = opponent(currentPlayer);
  const { evaluateFn, weights } = resolveEvaluator(options);

  const turns = generateTurns(state);
  if (turns.length === 0) return { turn: null, info: { elapsedMs: Date.now() - t0 } };

  // 即勝ちがあれば選ぶ
  for (const candidate of turns) {
    const nextState = {
      board: candidate.board,
      summonCounts: candidate.summonCounts,
      currentPlayer: opp,
      boardSize,
    };
    if (checkWinCondition(nextState, meta)) {
      return {
        turn: candidate,
        info: { elapsedMs: Date.now() - t0, score: evaluateFn(nextState, currentPlayer, weights) },
      };
    }
  }

  // ランダムに選ぶ
  const chosen = turns[Math.floor(Math.random() * turns.length)];
  const chosenNextState = {
    board: chosen.board,
    summonCounts: chosen.summonCounts,
    currentPlayer: opp,
    boardSize,
  };
  return {
    turn: chosen,
    info: { elapsedMs: Date.now() - t0, score: evaluateFn(chosenNextState, currentPlayer, weights) },
  };
}

/**
 * レベル2: 貪欲プレイヤー（即勝ち優先、次に evaluate スコア最大、同点ランダム）
 */
function level2Choose(state, options) {
  const t0 = Date.now();
  const { currentPlayer, boardSize } = state;
  const maxSummons = maxSummonsFor(boardSize);
  const meta = { maxSummons, boardSize };
  const opp = opponent(currentPlayer);
  const { evaluateFn, weights } = resolveEvaluator(options);

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
      return {
        turn: candidate,
        info: { elapsedMs: Date.now() - t0, score: evaluateFn(nextState, currentPlayer, weights) },
      };
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
    const score = evaluateFn(resultState, currentPlayer, weights);
    if (score > bestScore) {
      bestScore = score;
      bestCandidates = [candidate];
    } else if (score === bestScore) {
      bestCandidates.push(candidate);
    }
  }

  const chosen = bestCandidates[Math.floor(Math.random() * bestCandidates.length)];
  return { turn: chosen, info: { elapsedMs: Date.now() - t0, score: bestScore } };
}

/**
 * レベル3（将来 NNUE もここに合流予定）: 反復深化 α-β 探索。
 * options.maxDepth / options.timeBudgetMs でメインスレッドから調整可能。
 */
function level3Choose(state, options) {
  const maxDepth = (options && Number.isFinite(options.maxDepth)) ? options.maxDepth : 4;
  const timeBudgetMs = (options && Number.isFinite(options.timeBudgetMs)) ? options.timeBudgetMs : 1000;

  // TODO(Step5): options.net が指定されたら、ここで weights の代わりに
  // NNUE 評価関数を search.js に渡せるよう findBestTurn 側の拡張も必要になる。
  const { weights } = resolveEvaluator(options);

  const result = findBestTurn(state, {
    maxDepth,
    timeBudgetMs,
    weights,
    rng: Math.random,
  });
  return {
    turn: result.turn,
    info: {
      depthReached: result.depthReached,
      nodes: result.nodes,
      elapsedMs: result.elapsedMs,
      score: result.score,
    },
  };
}

// ───────────────────────── メッセージハンドラ ─────────────────────────

self.onmessage = async (event) => {
  const { requestId, state: rawState, level, options } = event.data;

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
    if (options && options.net) {
      // NNUE モデル指定時は level に関わらず NNUE 探索を行う。
      result = await nnueChoose(state, options);
    } else if (level === 1) {
      result = level1Choose(state, options);
    } else if (level === 3) {
      result = level3Choose(state, options);
    } else {
      // デフォルトはレベル2（ふつう）
      result = level2Choose(state, options);
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
