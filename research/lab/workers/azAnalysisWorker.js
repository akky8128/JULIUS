/**
 * azAnalysisWorker.js — 解析用 Web Worker（AlphaZero MCTS版）。
 *
 * analysisWorker.js（ab版）と同じメッセージ仕様・ラウンド制で、評価を MCTS
 * (analysisMcts.js の analyzeTopMovesMcts) に差し替えたもの。analysis.html は
 * netPath に genAZ*（方策ヘッド）を渡してこのワーカーを使う。
 *
 * ラウンドごとに総時間予算を倍増し、分岐数で割った「1候補あたり deadline」を MCTS に
 * 与えることで progressive に深く読む。stop / 新 analyze による上書きで打ち切り、
 * ラウンド境界での応答遅延は roundBudget 以下に有界化される（ab版と同じ設計）。
 *
 * 受信: { requestId, cmd:"analyze", state, netPath, topN? } / { requestId, cmd:"stop" }
 * 送信: { requestId, cmd:"progress", round, current, candidates, info } / "done" / "error"
 */

import { createNetworkFromBuffer } from "../../../core/nnue/network.js";
import { MctsAnalysisSession } from "../../../core/nnue/analysisMcts.js";

const DEFAULT_TOP_N = 3;
// 進捗の送信間隔（ミリ秒）。この間だけ木に sims を積んでから snapshot を送るため、
// 評価値・候補は実時間（ミリ秒）単位で滑らかに更新される。
const REPORT_INTERVAL_MS = 100;

// ── ネットのロード（Worker用・self.location.origin 基準の絶対URL）──
const netCache = new Map();
function loadNetForWorker(jsonPath) {
  const normalized = jsonPath.replace(/\.bin$/, ".json");
  let cached = netCache.get(normalized);
  if (cached) return cached;
  const jsonUrl = new URL(normalized, self.location.origin).href;
  const binPath = jsonUrl.replace(/\.json$/, ".bin");
  cached = Promise.all([
    fetch(jsonUrl).then((r) => { if (!r.ok) throw new Error(`モデルJSONの取得に失敗: ${normalized} (${r.status})`); return r.json(); }),
    fetch(binPath).then((r) => { if (!r.ok) throw new Error(`モデル重みの取得に失敗: ${binPath} (${r.status})`); return r.arrayBuffer(); }),
  ]).then(([meta, buf]) => createNetworkFromBuffer(buf, meta));
  netCache.set(normalized, cached);
  return cached;
}

// ── 停止フラグ・アクティブ管理 ──
const stopFlags = new Map();
let activeRequestId = 0;
const isStopRequested = (id) => stopFlags.get(id) === true;
const yieldToEventLoop = () => new Promise((resolve) => setTimeout(resolve, 0));

async function runAnalysis(requestId, state, netPath, topN) {
  const startTime = Date.now();
  const net = await loadNetForWorker(netPath);
  if (activeRequestId !== requestId) return;

  // 子局面ごとの MCTS 木を保持するセッションを構築（各候補の初期ネット評価もここで取得）。
  const session = new MctsAnalysisSession(state, net, { topN });
  let round = 0;

  while (activeRequestId === requestId && !isStopRequested(requestId)) {
    round++;
    // REPORT_INTERVAL_MS だけ木に sims を積む（上位候補に集中）。
    const sliceDeadline = Date.now() + REPORT_INTERVAL_MS;
    session.step(sliceDeadline);
    // 探索対象が無い（全候補が確定/合法手なし）等で step が即戻る場合も、progress を
    // ミリ秒spam しないようスライスを一定間隔に整える。
    const remain = sliceDeadline - Date.now();
    if (remain > 0) await new Promise((r) => setTimeout(r, remain));

    // キュー済みの stop / 新 analyze を処理させてから上書きを再確認。
    await yieldToEventLoop();
    if (activeRequestId !== requestId) break;

    const snap = session.snapshot();
    self.postMessage({
      requestId, cmd: "progress", round,
      current: snap.current, candidates: snap.candidates,
      info: { elapsedMs: Date.now() - startTime, totalCandidates: snap.totalCandidates },
    });

    if (snap.totalCandidates === 0) break; // 合法手なし（詰み）はこれ以上探索不要
    if (isStopRequested(requestId)) break;
  }

  self.postMessage({ requestId, cmd: "done", round, info: { elapsedMs: Date.now() - startTime } });
  stopFlags.delete(requestId);
}

self.onmessage = (event) => {
  const { requestId, cmd } = event.data;
  if (cmd === "stop") { stopFlags.set(requestId, true); return; }
  if (cmd === "analyze") {
    const { state, netPath, topN = DEFAULT_TOP_N } = event.data;
    activeRequestId = requestId;
    runAnalysis(requestId, state, netPath, topN).catch((err) => {
      stopFlags.delete(requestId);
      self.postMessage({ requestId, cmd: "error", error: err && err.message ? err.message : String(err) });
    });
    return;
  }
  self.postMessage({ requestId, cmd: "error", error: `未知の cmd: ${cmd}` });
};
