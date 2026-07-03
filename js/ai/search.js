/**
 * search.js — 反復深化ネガマックス α-β 探索モジュール（AI用）
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール。
 * Node.js 専用 API は一切使用しない（Date.now() でタイミング計測）。
 */

import { generateTurns, hashPosition } from "./moveGen.js";
import { checkWinCondition, maxSummonsFor } from "../gameLogic.js";
import { evaluate, WEIGHTS, WIN_SCORE } from "./evaluate.js";

// ───────────────────────── 定数 ─────────────────────────

/** 置換表エントリのフラグ */
const EXACT = 0;   // 正確なスコア
const LOWER = 1;   // 下限（α のカットオフ）
const UPPER = 2;   // 上限（β のカットオフ）

/** タイムチェック頻度（ノード数） */
const TIME_CHECK_INTERVAL = 256;

// ───────────────────────── ユーティリティ ─────────────────────────

/** プレイヤーを反転する */
function opponent(player) {
  return player === "white" ? "black" : "white";
}

/**
 * 候補手を 1-ply 評価でソートする（降順）。
 * 同点はシャッフル済みの順序を維持する（rng によるランダムタイブレーク）。
 *
 * @param {Array<{actions, board, summonCounts}>} candidates
 * @param {string} currentPlayer
 * @param {number} boardSize
 * @param {typeof WEIGHTS} weights
 * @returns {Array<{actions, board, summonCounts, _score}>}
 */
function sortCandidates(candidates, currentPlayer, boardSize, weights) {
  const opp = opponent(currentPlayer);
  return candidates
    .map(cand => {
      const childState = {
        board: cand.board,
        summonCounts: cand.summonCounts,
        currentPlayer: opp,
        boardSize,
      };
      // ネガマックス規約: 現在のプレイヤー視点で評価
      const score = evaluate(childState, currentPlayer, weights);
      return { ...cand, _score: score };
    })
    .sort((a, b) => b._score - a._score);
}

// ───────────────────────── 主要エクスポート ─────────────────────────

/**
 * 反復深化ネガマックス α-β 探索で最善手を返す。
 *
 * @param {{ board: Array[][], summonCounts: {white:number,black:number},
 *            currentPlayer: string, boardSize: number }} state
 *   board は正規化済みであること。
 * @param {{
 *   maxDepth?: number,
 *   timeBudgetMs?: number,
 *   weights?: typeof WEIGHTS,
 *   rng?: () => number
 * }} options
 * @returns {{
 *   turn: {actions: object[], board: Array[][], summonCounts: {white:number,black:number}} | null,
 *   score: number,
 *   depthReached: number,
 *   nodes: number,
 *   elapsedMs: number
 * }}
 */
export function findBestTurn(state, options = {}) {
  const {
    maxDepth = 4,
    timeBudgetMs = 1000,
    weights = WEIGHTS,
    rng = Math.random,
  } = options;

  const { currentPlayer, boardSize } = state;
  const maxSummons = maxSummonsFor(boardSize);
  const meta = { maxSummons, boardSize };

  const startTime = Date.now();
  let nodeCount = 0;
  let timedOut = false;

  // 置換表（findBestTurn 呼び出しごとに新規作成）
  // キー: hashPosition(board, summonCounts) + "|" + currentPlayer
  // 値: { depth, score, flag, bestCandHash }
  const tt = new Map();

  /**
   * タイムアウトチェック（nodeCount が CHECK_INTERVAL の倍数のときだけ実施）
   */
  function checkTime() {
    if (Date.now() - startTime >= timeBudgetMs) {
      timedOut = true;
    }
  }

  /**
   * ネガマックス α-β 探索の本体（再帰）。
   *
   * @param {{ board, summonCounts, currentPlayer, boardSize }} curState
   * @param {number} depth  - 残り探索深さ（0 でリーフ評価）
   * @param {number} alpha
   * @param {number} beta
   * @param {number} ply    - ルートからの手数（勝ち/負けスコアの調整用）
   * @returns {number}  現在のプレイヤー視点のスコア
   */
  function negamax(curState, depth, alpha, beta, ply) {
    nodeCount++;

    // タイムチェック（一定間隔）
    if (nodeCount % TIME_CHECK_INTERVAL === 0) {
      checkTime();
    }
    if (timedOut) return 0; // 中断: 0 を返す（この深さの結果は破棄される）

    const { board, summonCounts, boardSize: bs } = curState;
    const player = curState.currentPlayer;

    // ── 置換表参照 ──
    const posKey = hashPosition(board, summonCounts) + "|" + player;
    const ttEntry = tt.get(posKey);
    let ttBestCandHash = null;

    if (ttEntry && ttEntry.depth >= depth) {
      const { score: ttScore, flag } = ttEntry;
      if (flag === EXACT) return ttScore;
      if (flag === LOWER && ttScore > alpha) alpha = ttScore;
      if (flag === UPPER && ttScore < beta)  beta  = ttScore;
      if (alpha >= beta) return ttScore;
    }
    if (ttEntry) ttBestCandHash = ttEntry.bestCandHash;

    // ── 現在のプレイヤーに合法手がなければ負け ──
    const candidates = generateTurns(curState);
    if (candidates.length === 0) {
      // 現在のプレイヤーに手がない → このプレイヤーの負け
      return -(WIN_SCORE - ply);
    }

    // ── 深さ 0 ならリーフ評価 ──
    if (depth === 0) {
      return evaluate(curState, player, weights);
    }

    // ── 手順並べ替え ──
    // 1. 置換表の最善手を先頭に
    // 2. エリミネートを優先
    // 3. 残りはシャッフル済み（呼び出し前にシャッフル済み想定）
    //
    // root では rng シャッフル後に 1-ply ソートするが、
    // 内部ノードはコスト削減のため TT ベスト手 + エリミネート優先のみ行う。
    let orderedCandidates = candidates;

    if (ttBestCandHash !== null) {
      // TT の最善手を先頭に
      const ttIdx = candidates.findIndex(c =>
        hashPosition(c.board, c.summonCounts) + "|" + opponent(player) === ttBestCandHash
      );
      if (ttIdx > 0) {
        orderedCandidates = [candidates[ttIdx], ...candidates.slice(0, ttIdx), ...candidates.slice(ttIdx + 1)];
      }
    } else {
      // エリミネート（actions[0].type === "eliminate"）を先頭に
      const elims = [];
      const rest  = [];
      for (const c of candidates) {
        if (c.actions[0]?.type === "eliminate") elims.push(c);
        else rest.push(c);
      }
      orderedCandidates = [...elims, ...rest];
    }

    // ── 子ノード展開 ──
    let bestScore = -Infinity;
    let bestCandHash = null;
    const origAlpha = alpha;

    for (const cand of orderedCandidates) {
      if (timedOut) break;

      const opp = opponent(player);

      // 候補を適用した後の状態
      const childState = {
        board: cand.board,
        summonCounts: cand.summonCounts,
        currentPlayer: opp,
        boardSize: bs,
      };

      // ── 即勝ちチェック: 相手に合法手がなければ +WIN ──
      // （子ノードに入る前に確認することでカットを早める）
      const childMeta = { maxSummons, boardSize: bs };
      if (checkWinCondition(childState, childMeta)) {
        // 相手が手なし → 現在のプレイヤーの勝ち
        const score = WIN_SCORE - ply;
        bestScore = score;
        bestCandHash = hashPosition(cand.board, cand.summonCounts) + "|" + opp;
        // 置換表に記録して即リターン（これ以上良い手はない）
        tt.set(posKey, { depth, score, flag: EXACT, bestCandHash });
        return score;
      }

      // ── 再帰 ──
      const childScore = -negamax(childState, depth - 1, -beta, -alpha, ply + 1);

      if (timedOut) break;

      if (childScore > bestScore) {
        bestScore = childScore;
        bestCandHash = hashPosition(cand.board, cand.summonCounts) + "|" + opp;
      }
      if (childScore > alpha) alpha = childScore;
      if (alpha >= beta) break; // β カット
    }

    if (timedOut) return bestScore === -Infinity ? 0 : bestScore;

    // ── 置換表に記録 ──
    let flag;
    if (bestScore <= origAlpha)    flag = UPPER;
    else if (bestScore >= beta)    flag = LOWER;
    else                           flag = EXACT;

    tt.set(posKey, { depth, score: bestScore, flag, bestCandHash });

    return bestScore;
  }

  // ─── 反復深化ループ ───────────────────────────────────────────

  let bestTurn  = null;
  let bestScore = -Infinity;
  let depthReached = 0;

  for (let depth = 1; depth <= maxDepth; depth++) {
    if (timedOut) break;

    // ルートの候補を生成してシャッフル（タイブレーク用）
    const rootCandidates = generateTurns(state);
    if (rootCandidates.length === 0) break;

    // Fisher-Yates シャッフル（rng によるランダムタイブレーク）
    for (let i = rootCandidates.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [rootCandidates[i], rootCandidates[j]] = [rootCandidates[j], rootCandidates[i]];
    }

    // ルートでは 1-ply ソートも行う（move ordering の効果大）
    const sortedRoot = sortCandidates(rootCandidates, currentPlayer, boardSize, weights);

    let depthBestTurn  = null;
    let depthBestScore = -Infinity;
    let alpha = -Infinity;
    const beta = Infinity;

    for (const cand of sortedRoot) {
      if (timedOut) break;

      const opp = opponent(currentPlayer);
      const childState = {
        board: cand.board,
        summonCounts: cand.summonCounts,
        currentPlayer: opp,
        boardSize,
      };

      // 即勝ちチェック
      if (checkWinCondition(childState, meta)) {
        depthBestTurn  = cand;
        depthBestScore = WIN_SCORE - depth;
        break; // 即勝ちなら探索終了
      }

      const score = -negamax(childState, depth - 1, -beta, -alpha, 1);

      if (timedOut) {
        // タイムアウト: この手は未評価かもしれないが、
        // depthBestTurn が null の場合はこの手を暫定として採用
        if (depthBestTurn === null) {
          depthBestTurn  = cand;
          depthBestScore = score;
        }
        break;
      }

      if (score > depthBestScore) {
        depthBestScore = score;
        depthBestTurn  = cand;
        if (score > alpha) alpha = score;
      }
    }

    // タイムアウトの場合はこの深さの結果を破棄（ただし depth=1 は常に採用）
    if (!timedOut || depth === 1) {
      if (depthBestTurn !== null) {
        bestTurn     = depthBestTurn;
        bestScore    = depthBestScore;
        depthReached = depth;
      }
    }

    // 即勝ちが見つかれば深化不要
    if (depthBestScore >= WIN_SCORE - maxDepth * 2) break;
  }

  // 候補がまったくない場合
  if (bestTurn === null) {
    const rootCands = generateTurns(state);
    if (rootCands.length > 0) {
      bestTurn  = rootCands[0];
      bestScore = -Infinity;
    }
  }

  const elapsedMs = Date.now() - startTime;

  return {
    turn:         bestTurn,
    score:        bestScore,
    depthReached,
    nodes:        nodeCount,
    elapsedMs,
  };
}
