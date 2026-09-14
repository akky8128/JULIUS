/**
 * analysisMcts.js — 局面解析コア（MCTS版）。analysis.js(analyzeTopMoves) の
 * ab(findBestTurn) を AlphaZero の MCTS 評価に差し替えたもの。progress の返り値形状は
 * analysis.js と互換（current / candidates）に保ち、解析UI(analysis.html)を最小改変で使える。
 *
 * 各合法ターン候補 c について、その子局面（相手手番の TURN_START）から MCTS を回し、
 * 相手視点の root 価値を反転して現手番視点スコア（[-1,1]）とする。時間予算は deadlineMs
 * を1候補あたりに与える（呼び出し側がラウンドごとに増やして progressive に深くする）。
 *
 * ブラウザ/Node 両対応の純 ESM（DOM/Worker 非依存）。tools/alphazero の MCTS 資産を流用。
 */

import { normalizeBoard, maxSummonsFor, checkWinCondition } from "../../gameLogic.js";
import { generateTurns } from "../moveGen.js";
import { diffToNotation } from "../../tutorial/coords.js";
import { makeRootNode, runMCTS, simulate, expandNode, DEFAULT_C_PUCT } from "../../../tools/alphazero/mcts.mjs";

function opponent(p) { return p === "white" ? "black" : "white"; }

/** root の合法手の訪問数重み付き平均 Q（root.mover 視点 [-1,1]）。未訪問時は 0。 */
function rootValueMover(root) {
  let sumN = 0, sumW = 0;
  for (const a of root.legal) {
    if (root.N[a] > 0) { sumN += root.N[a]; sumW += root.W[a]; }
  }
  return sumN > 0 ? sumW / sumN : 0;
}

/**
 * 指定局面の上位候補手を MCTS 評価で解析する（analyze.js と同じ返り値形状）。
 * @param {object} state - { board, summonCounts, currentPlayer, boardSize }
 * @param {object} net - createNetwork 由来（evaluatePolicy を持つ AZ ネット）
 * @param {{ topN?, deadlineMs?, simsPerChild?, cPuct?, rng? }} [options]
 * @returns {{ current:{score,winProb,whiteWinProb}, totalCandidates:number, candidates:Array }}
 */
export function analyzeTopMovesMcts(state, net, options = {}) {
  const { topN = 3, deadlineMs = 120, simsPerChild = null, cPuct, rng = Math.random } = options;
  const { boardSize, currentPlayer, summonCounts } = state;
  const normalized = normalizeBoard(state.board, boardSize);
  const nstate = { board: normalized, summonCounts, currentPlayer, boardSize };
  const meta = { maxSummons: maxSummonsFor(boardSize), boardSize };
  const opp = opponent(currentPlayer);

  const cands = generateTurns(nstate);
  if (cands.length === 0) {
    return { current: { score: 0, winProb: 0.5, whiteWinProb: 0.5 }, totalCandidates: 0, candidates: [] };
  }

  const scored = cands.map((c) => {
    const childState = { board: c.board, summonCounts: c.summonCounts, currentPlayer: opp, boardSize };
    let score, isWin = false;
    if (checkWinCondition(childState, meta)) {
      score = 1; isWin = true; // この手で現手番が勝ち
    } else {
      const root = makeRootNode(childState); // 相手手番の TURN_START
      if (root.isTerminal) {
        score = 1; isWin = true; // 相手に合法手なし = 現手番の勝ち
      } else {
        const opts = { minSims: 8 };
        if (simsPerChild != null) opts.simulations = simsPerChild;
        else opts.deadline = Date.now() + deadlineMs;
        if (cPuct != null) opts.cPuct = cPuct;
        runMCTS(root, net, opts);
        score = -rootValueMover(root); // 相手視点価値を反転 → 現手番視点 [-1,1]
      }
    }
    let notation = diffToNotation(normalized, c.board, boardSize);
    if (isWin) notation += "#";
    return {
      actions: c.actions, board: c.board, summonCounts: c.summonCounts,
      notation, score, winProb: (score + 1) / 2, isWin,
    };
  });

  // score 降順（安定ソート: 同点は生成順維持）
  scored.sort((a, b) => b.score - a.score);
  const candidates = scored.slice(0, topN);

  const currentScore = candidates.length ? candidates[0].score : 0;
  const winProb = (currentScore + 1) / 2;                       // 現手番勝率
  const whiteScore = currentPlayer === "white" ? currentScore : -currentScore;
  const whiteWinProb = (whiteScore + 1) / 2;                    // 白視点勝率（評価バー用）
  return { current: { score: currentScore, winProb, whiteWinProb }, totalCandidates: cands.length, candidates };
}

// ───────────────────────── 継続セッション（ミリ秒単位更新用）─────────────────────────
/**
 * MctsAnalysisSession — 局面の各合法ターン子局面ごとに MCTS 木を「永続保持」し、少しずつ
 * sims を積みながら任意タイミングで snapshot（current + candidates）を取り出せる解析セッション。
 *
 * ワーカーが短い時間スライス（例 100ms）ごとに step→snapshot→postMessage することで、
 * 「1ラウンド全候補評価の完了待ち」ではなく実時間（ミリ秒）単位で評価・候補が更新される。
 * sims は木に蓄積され続けるため、時間とともに評価は滑らかに収束する（nnue-lab と同様）。
 *
 * 計算は上位 topK 候補に集中させ、下位候補は初期のネット評価値（expandNode の value）を
 * 保持する（無駄な探索と木のメモリ増加を避ける）。表示は topN 候補。
 */
export class MctsAnalysisSession {
  constructor(state, net, { topN = 3, topK = 16, cPuct = DEFAULT_C_PUCT, batch = 4 } = {}) {
    this.net = net;
    this.topN = topN;
    this.topK = topK;
    this.cPuct = cPuct;
    this.batch = batch;
    const { boardSize, currentPlayer, summonCounts } = state;
    this.boardSize = boardSize;
    this.currentPlayer = currentPlayer;
    this.normalized = normalizeBoard(state.board, boardSize);
    const nstate = { board: this.normalized, summonCounts, currentPlayer, boardSize };
    const meta = { maxSummons: maxSummonsFor(boardSize), boardSize };
    const opp = opponent(currentPlayer);

    const cands = generateTurns(nstate);
    this.totalCandidates = cands.length;
    // 各候補: 子局面の root を作り、即時に expandNode で初期評価（現手番視点スコア）を得る。
    // これにより最初の snapshot からネット評価順の候補が並ぶ（数十ms）。
    this.children = cands.map((c) => {
      const childState = { board: c.board, summonCounts: c.summonCounts, currentPlayer: opp, boardSize };
      let terminalScore = null;
      let root = null;
      let initScore = 0;
      if (checkWinCondition(childState, meta)) {
        terminalScore = 1; // この手で現手番が勝ち
      } else {
        root = makeRootNode(childState);
        if (root.isTerminal) { terminalScore = 1; root = null; }
        else { initScore = -expandNode(root, net); } // 相手視点value → 現手番スコア
      }
      let notation = diffToNotation(this.normalized, c.board, boardSize);
      if (terminalScore !== null) notation += "#";
      return { cand: c, root, terminalScore, initScore, notation };
    });
  }

  /** 子候補の現在スコア（現手番視点 [-1,1]）。訪問済みは root 平均Q、未訪問は初期ネット評価。 */
  _scoreOf(ch) {
    if (ch.terminalScore !== null) return ch.terminalScore;
    if (ch.root && ch.root.Nnode > 0) return -rootValueMover(ch.root);
    return ch.initScore;
  }

  /** deadlineMs（絶対時刻）まで、現在スコア上位 topK の子に sims を積む。 */
  step(deadlineMs) {
    const active = this.children.filter((ch) => ch.terminalScore === null && ch.root);
    if (active.length === 0) return;
    active.sort((a, b) => this._scoreOf(b) - this._scoreOf(a));
    const pool = active.slice(0, this.topK);
    let i = 0;
    while (Date.now() < deadlineMs) {
      const ch = pool[i % pool.length];
      for (let s = 0; s < this.batch; s++) simulate(ch.root, this.net, this.cPuct);
      i++;
    }
  }

  /** 現時点の解析結果（analyzeTopMovesMcts と同一形状）を返す。 */
  snapshot() {
    const scored = this.children.map((ch) => {
      const score = this._scoreOf(ch);
      return {
        actions: ch.cand.actions, board: ch.cand.board, summonCounts: ch.cand.summonCounts,
        notation: ch.notation, score, winProb: (score + 1) / 2, isWin: ch.terminalScore !== null,
      };
    });
    scored.sort((a, b) => b.score - a.score);
    const candidates = scored.slice(0, this.topN);
    const currentScore = candidates.length ? candidates[0].score : 0;
    const whiteScore = this.currentPlayer === "white" ? currentScore : -currentScore;
    return {
      current: { score: currentScore, winProb: (currentScore + 1) / 2, whiteWinProb: (whiteScore + 1) / 2 },
      totalCandidates: this.totalCandidates,
      candidates,
    };
  }
}
