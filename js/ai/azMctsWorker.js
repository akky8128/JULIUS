/**
 * azMctsWorker.js — AlphaZero(方策ヘッド)ネットを MCTS+マイクロアクションで「1ターン指す」
 * ブラウザ Worker。nnue-lab.html の CPU 対戦/観戦で genAZ* ネットを本来の強さで動かす。
 *
 * nnueWorker.js と互換のリクエスト/レスポンス:
 *   in : { requestId, state, options:{ timeBudgetMs?, simulations?, net } }
 *   out(結果): { requestId, turn:{ actions, board, summonCounts }|null, info }
 *   out(思考中): { requestId, type:'eval', whiteWinProb, info }   ← 評価バー動的更新用
 *   out(異常): { requestId, error }
 *
 * ★強さの核心（analysis.html/nnue-lab の従来αβとの違い）★
 *   価値ヘッド単体のαβ(ターン文脈ゼロ)ではなく、マイクロ状態ごとに turn-context を与えた
 *   MCTS で召喚/排除の手順を探索する。tools/alphazero の学習資産をそのまま流用する
 *   （モジュールは絶対パス /tools/alphazero/... で読み込み、内部の相対 import は自動解決）。
 */

import { normalizeBoard } from "../gameLogic.js";
import { generateTurns, hashPosition } from "./moveGen.js";
import { createNetworkFromBuffer } from "./nnue/network.js";
import { initTurnState, applyMicroAction } from "../../tools/alphazero/microActions.mjs";
import {
  makeNode,
  expandNode,
  simulate,
  bestActionByVisits,
  viableActions,
  DEFAULT_C_PUCT,
} from "../../tools/alphazero/mcts.mjs";

// ───────────────────────── ネットのロード（キャッシュ付き）─────────────────────────
const netCache = new Map(); // path(.json) -> Promise<network>
function loadNet(jsonPath) {
  const normalized = jsonPath.replace(/\.bin$/, ".json");
  let cached = netCache.get(normalized);
  if (cached) return cached;
  const jsonUrl = new URL(normalized, location.origin).href;
  const binUrl = jsonUrl.replace(/\.json$/, ".bin");
  cached = Promise.all([
    fetch(jsonUrl).then((r) => { if (!r.ok) throw new Error(`モデルJSON取得失敗: ${normalized} (${r.status})`); return r.json(); }),
    fetch(binUrl).then((r) => { if (!r.ok) throw new Error(`モデル重み取得失敗: ${binUrl} (${r.status})`); return r.arrayBuffer(); }),
  ]).then(([meta, buf]) => createNetworkFromBuffer(buf, meta));
  netCache.set(normalized, cached);
  return cached;
}

// ───────────────────────── 評価値（root の訪問数重み付き平均 Q）─────────────────────────
/** root の合法手の訪問数重み付き平均 Q（mover 視点 [-1,1]）。未訪問時は 0。 */
function rootValueMover(root) {
  let sumN = 0, sumW = 0;
  for (const a of root.legal) {
    if (root.N[a] > 0) { sumN += root.N[a]; sumW += root.W[a]; }
  }
  return sumN > 0 ? sumW / sumN : 0;
}

/** mover 視点勝率 → 白視点勝率 [0,1] に変換。 */
function toWhiteWinProb(valueMover, mover) {
  const pMover = (valueMover + 1) / 2;
  return mover === "white" ? pMover : 1 - pMover;
}

// ───────────────────────── 1ターン探索（eval ストリーミング付き）─────────────────────────
/**
 * playTurnMcts（tools/alphazero/mctsPlayer.mjs）と同一意味論だが、各マイクロ決定の探索を
 * チャンク実行して root 評価値を逐次 report(whiteWinProb) する。返り値は最終 ctx。
 */
function playTurnStreaming(state, net, opts, report) {
  const { timeBudgetMs = 1000, simulations = null, cPuct = DEFAULT_C_PUCT, minSims = 8, maxMicroSteps = 64, chunk = 16 } = opts;
  const boardSize = state.boardSize;
  const mover = state.currentPlayer;
  let ctx = initTurnState(state);
  const microActions = [];
  let totalSims = 0;
  // ターン全体で共有する deadline（後続マイクロ手にも時間を残す）。
  const turnDeadline = simulations == null ? Date.now() + timeBudgetMs : null;

  for (let step = 0; step < maxMicroSteps; step++) {
    const legal = viableActions(ctx);
    if (legal.length === 0) throw new Error(`azMcts: no viable micro-action at step ${step} (phase=${ctx.phase})`);
    if (legal.length === 1) {
      ctx = applyMicroAction(ctx, legal[0]);
      microActions.push(legal[0]);
      if (ctx.done) break;
      continue;
    }

    const root = makeNode(ctx, mover, boardSize);
    // 事前展開して即時に初期評価を report（探索開始直後から評価バーが動く）。
    expandNode(root, net);
    let sims = 0;
    for (;;) {
      const budgetLeft = simulations != null ? simulations - sims : Infinity;
      const n = Math.min(chunk, budgetLeft);
      for (let i = 0; i < n; i++) { simulate(root, net, cPuct); sims++; }
      report(toWhiteWinProb(rootValueMover(root), mover), totalSims + sims);
      if (simulations != null && sims >= simulations) break;
      if (turnDeadline != null && sims >= minSims && Date.now() >= turnDeadline) break;
      if (n === 0) break;
    }
    totalSims += sims;
    const action = bestActionByVisits(root, { temperature: 0 });
    ctx = applyMicroAction(ctx, action);
    microActions.push(action);
    if (ctx.done) break;
  }
  if (!ctx.done) throw new Error("azMcts: turn did not terminate within maxMicroSteps");
  return { ctx, totalSims, microSteps: microActions.length };
}

// ───────────────────────── MCTS 最終盤面 → generateTurns 候補(actions) 照合 ─────────────────────────
function matchCandidate(state, finalBoard, finalSummonCounts) {
  const size = state.boardSize;
  const targetHash = hashPosition(normalizeBoard(finalBoard, size), finalSummonCounts);
  const cands = generateTurns(state);
  for (const c of cands) {
    if (hashPosition(normalizeBoard(c.board, size), c.summonCounts) === targetHash) return c;
  }
  return null;
}

// ───────────────────────── メッセージ処理 ─────────────────────────
self.onmessage = async (event) => {
  const { requestId, state, options = {} } = event.data || {};
  try {
    const net = await loadNet(options.net);
    const size = state.boardSize;
    const normState = {
      board: normalizeBoard(state.board, size),
      summonCounts: { ...state.summonCounts },
      currentPlayer: state.currentPlayer,
      boardSize: size,
    };
    if (generateTurns(normState).length === 0) {
      self.postMessage({ requestId, turn: null, info: { reason: "no-legal-turn" } });
      return;
    }

    const t0 = performance.now();
    let lastWhiteWinProb = 0.5;
    const report = (whiteWinProb, sims) => {
      lastWhiteWinProb = whiteWinProb;
      self.postMessage({ requestId, type: "eval", whiteWinProb, info: { sims } });
    };

    const { ctx, totalSims, microSteps } = playTurnStreaming(normState, net, {
      timeBudgetMs: options.timeBudgetMs,
      simulations: options.simulations ?? null,
    }, report);

    const cand = matchCandidate(normState, ctx.board, ctx.summonCounts);
    const elapsedMs = performance.now() - t0;
    if (!cand) {
      self.postMessage({ requestId, error: "azMcts: MCTS結果に一致する合法ターンが見つかりません" });
      return;
    }
    self.postMessage({
      requestId,
      turn: { actions: cand.actions, board: cand.board, summonCounts: cand.summonCounts },
      info: { engine: "az-mcts", nodes: totalSims, microSteps, elapsedMs, whiteWinProb: lastWhiteWinProb },
    });
  } catch (err) {
    self.postMessage({ requestId, error: (err && err.message) || String(err) });
  }
};
