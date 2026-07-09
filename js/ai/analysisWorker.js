/**
 * analysisWorker.js — 解析用 Web Worker（Phase 2: 反復深化マルチPV探索）
 *
 * ブラウザの Web Worker として動作する ESM モジュール。
 * js/ai/nnue/analysis.js の analyzeTopMoves（純関数）を、深さ d=1,2,3,...
 * と反復深化させながら繰り返し呼び出し、深さが確定するごとに部分結果を
 * メインスレッドへ progress として逐次送信する。
 *
 * NNUE ネットワークのロードは js/ai/nnueWorker.js の loadNetForWorker と
 * 同じ方式（fetch + createNetworkFromBuffer、パスごとに Promise キャッシュ）
 * を踏襲する。Worker 内には window が存在せず、network.js の loadNetwork()
 * は window の有無で環境判定するため誤って Node 経路（fs）に落ちてしまう。
 * そのため loadNetwork() は使わず、本ファイルで独自にロード処理を持つ。
 *
 * ── メッセージ仕様 ──────────────────────────────────────────────
 *
 * 受信（解析開始）:
 *   {
 *     requestId: number,
 *     cmd: "analyze",
 *     state: { board, summonCounts, currentPlayer, boardSize },
 *     netPath: string,        // 例 "models/gen071.json"（.bin は自動導出）
 *     maxDepth?: number,      // 既定 64（実質「時間無制限＝停止まで」）
 *     timeBudgetMs?: number,  // 未使用（各深さは内部で十分大きい予算を使う）
 *     topN?: number,          // 既定 3
 *   }
 *
 * 受信（停止要求）:
 *   { requestId: number, cmd: "stop" }
 *   → 進行中の反復深化ループを「次の深さ境界」で止める。
 *     findBestTurn は深さの途中では中断できないため、深さ境界でのみ確認する。
 *
 * 送信（進捗・深さ d が完了するたび）:
 *   {
 *     requestId,
 *     cmd: "progress",
 *     depth: number,
 *     current: { score, winProb, whiteWinProb },
 *     candidates: Array<{ actions, board, summonCounts, notation, score, winProb, isWin }>,
 *     info: { elapsedMs: number },
 *   }
 *
 * 送信（完了）:
 *   { requestId, cmd: "done", depth: number, info: { elapsedMs: number } }
 *
 * 送信（エラー）:
 *   { requestId, cmd: "error", error: string }
 */

import { createNetworkFromBuffer } from "./nnue/network.js";
import { analyzeTopMoves } from "./nnue/analysis.js";

// ───────────────────────── 定数 ─────────────────────────

/** findBestTurn に渡す探索深さの上限。実際の打ち切りは時間予算側で行う。 */
const DEFAULT_MAX_DEPTH = 64;
/** 既定の上位候補数。 */
const DEFAULT_TOP_N = 3;

// ── ラウンド制の時間予算 ──
// 解析は「ラウンド」を繰り返し、ラウンドごとに総時間予算を倍増させて
// 徐々に深く読む（progressive）。各ラウンドの総時間は分岐数で分割して
// 1手あたりの findBestTurn 予算にするため、1ラウンドの実時間は
// おおよそ roundBudget に収まり、stop の応答遅延もそれ以下に有界化される。
// （findBestTurn は1手の探索を途中で中断できないため、応答性は
//  「1手あたり予算 × 分岐数 ≒ roundBudget」で決まる。）

/** 初回ラウンドの総時間予算（分岐数が判明する前）。 */
const INITIAL_ROUND_BUDGET_MS = 500;
/** ラウンド総予算の上限（stop の最悪応答遅延をこの程度に抑える）。 */
const MAX_ROUND_BUDGET_MS = 8000;
/** 分岐数が未知の初回ラウンドで用いる1手あたり予算。 */
const INITIAL_PER_CHILD_MS = 120;
/** 1手あたり予算の下限（分岐が多い局面で過小にならないように）。 */
const MIN_PER_CHILD_MS = 60;

// ───────────────────────── NNUE ネットワークのロード（Worker用） ─────────────────────────
//
// nnueWorker.js の loadNetForWorker と同じ方式。self.location.origin を基準に
// 絶対URL化してから fetch する（相対パスは Worker スクリプト自身のディレクトリ
// 基準で解決されてしまうため）。

/** @type {Map<string, Promise<ReturnType<typeof createNetworkFromBuffer>>>} */
const netCache = new Map();

/**
 * netPath（.json）から NNUE network をロードする（キャッシュ付き）。
 * @param {string} jsonPath - 例 "models/gen071.json"（.bin は自動導出）
 * @returns {Promise<ReturnType<typeof createNetworkFromBuffer>>}
 */
function loadNetForWorker(jsonPath) {
  const normalized = jsonPath.replace(/\.bin$/, ".json");
  let cached = netCache.get(normalized);
  if (cached) return cached;

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

// ───────────────────────── 停止フラグ管理 ─────────────────────────

/** requestId ごとの停止フラグ。深さ境界でのみ参照される。 */
const stopFlags = new Map();

/** requestId の停止を要求する。 */
function requestStop(requestId) {
  stopFlags.set(requestId, true);
}

/** requestId が停止要求済みかどうか。 */
function isStopRequested(requestId) {
  return stopFlags.get(requestId) === true;
}

/** requestId の停止フラグを解放する（解析終了時に呼ぶ）。 */
function clearStop(requestId) {
  stopFlags.delete(requestId);
}

// ───────────────────────── 解析ループ本体 ─────────────────────────

/**
 * 現在アクティブな解析の requestId。新しい analyze を受信するたびに更新され、
 * 進行中の古いループは深さ境界でこれと比較して自分が上書きされたことを検出し打ち切る。
 * （メインスレッド側は requestId 不一致の progress を無視するが、Worker 側でも
 *  旧ループを止めないと無駄な深い探索が走り続け CPU を占有してしまうため。）
 */
let activeRequestId = 0;

/** 次のマクロタスクまでイベントループに制御を返す（キュー済みの stop/analyze を処理させる）。 */
function yieldToEventLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * ラウンド制のマルチPV解析。ラウンドごとに総時間予算を倍増させながら
 * analyzeTopMoves を繰り返し、そのつど progress を送信する。stop 受信・
 * 新しい解析による上書き（activeRequestId 変化）で終了して done を送信する。
 *
 * 各ラウンドの総予算は分岐数で分割して1手あたりの findBestTurn 予算とするため、
 * 1ラウンドの実時間は概ね roundBudget（上限 MAX_ROUND_BUDGET_MS）に収まり、
 * ラウンド境界での stop 応答遅延もそれ以下に有界化される。
 *
 * @param {number} requestId
 * @param {object} state - { board, summonCounts, currentPlayer, boardSize }
 * @param {string} netPath
 * @param {number} maxDepth - findBestTurn の探索深さ上限（実際は時間予算で打ち切る）
 * @param {number} topN
 */
async function runAnalysis(requestId, state, netPath, maxDepth, topN) {
  const startTime = Date.now();
  const net = await loadNetForWorker(netPath);
  if (activeRequestId !== requestId) return; // ロード待ちの間に上書きされた

  let roundBudget = INITIAL_ROUND_BUDGET_MS;
  let totalCandidates = null; // 初回ラウンドで判明する
  let round = 0;

  while (activeRequestId === requestId && !isStopRequested(requestId)) {
    round++;

    // 分岐数が分かっていれば総予算を分割、未知（初回）は控えめな既定値を使う。
    const perChild = totalCandidates
      ? Math.max(MIN_PER_CHILD_MS, Math.floor(roundBudget / totalCandidates))
      : INITIAL_PER_CHILD_MS;

    const res = analyzeTopMoves(state, net, {
      maxDepth,
      timeBudgetMs: perChild,
      topN,
      rng: Math.random,
    });
    totalCandidates = res.totalCandidates;

    // このラウンドの計算中にキュー済みだった stop / 新しい analyze を処理させるため、
    // progress 送信前に一度イベントループへ制御を返す。戻ったら上書きを再確認する。
    await yieldToEventLoop();
    if (activeRequestId !== requestId) break;

    self.postMessage({
      requestId,
      cmd: "progress",
      round,
      current: res.current,
      candidates: res.candidates,
      info: {
        elapsedMs: Date.now() - startTime,
        roundBudgetMs: roundBudget,
        perChildBudgetMs: perChild,
        totalCandidates,
      },
    });

    // 合法手がない（詰み/手詰まり）場合はこれ以上探索しても無意味なので終了する。
    if (res.candidates.length === 0) break;
    if (isStopRequested(requestId)) break;

    // 次ラウンドは総予算を倍増（上限まで）。上限到達後も stop まで探索を継続する。
    roundBudget = Math.min(MAX_ROUND_BUDGET_MS, roundBudget * 2);
  }

  self.postMessage({
    requestId,
    cmd: "done",
    round,
    info: { elapsedMs: Date.now() - startTime },
  });

  clearStop(requestId);
}

// ───────────────────────── メッセージハンドラ ─────────────────────────

self.onmessage = (event) => {
  const { requestId, cmd } = event.data;

  if (cmd === "stop") {
    requestStop(requestId);
    return;
  }

  if (cmd === "analyze") {
    const {
      state,
      netPath,
      maxDepth = DEFAULT_MAX_DEPTH,
      topN = DEFAULT_TOP_N,
    } = event.data;

    // このリクエストをアクティブにする。進行中の古いループは次の深さ境界で
    // activeRequestId の変化を検出して自動的に打ち切られる。
    activeRequestId = requestId;

    runAnalysis(requestId, state, netPath, maxDepth, topN).catch((err) => {
      clearStop(requestId);
      self.postMessage({
        requestId,
        cmd: "error",
        error: err && err.message ? err.message : String(err),
      });
    });
    return;
  }

  self.postMessage({
    requestId,
    cmd: "error",
    error: `未知の cmd: ${cmd}`,
  });
};
