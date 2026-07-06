/**
 * search.js — 反復深化ネガマックス α-β 探索モジュール（AI用）
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール。
 * Node.js 専用 API は一切使用しない（Date.now() でタイミング計測）。
 */

import { generateTurns, hashPosition } from "./moveGen.js";
import { checkWinCondition, maxSummonsFor } from "../gameLogic.js";
import { evaluate, WEIGHTS, WIN_SCORE } from "./evaluate.js";
import { createZobrist, hashFromScratch, hashToKey } from "./zobrist.js";
import { applyTurn } from "./searchGen.js";

// NOTE:
// generateTurns() (moveGen.js, 変更禁止) は候補ごとに結果局面の board を
// フルクローンして返す。これは列挙の「正しさ」を担保する唯一の実装であり、
// 本ファイルはその結果集合を変えずに使い続ける。
// 一方、探索木を下る際に candidate.board をそのまま子ノードとして使うと
// 深い探索でクローンが積み重なる。そこで、探索の recursion では
// 「1つの可変ボード (searchBoard) に candidate.actions を適用 → 再帰 → 元に戻す」
// という make/unmake 方式に切り替え、フルクローンを避ける。
// TT キーは文字列 hashPosition の代わりに Zobrist ハッシュ (hi/lo 差分更新) を使う。

// ───────────────────────── 定数 ─────────────────────────

/** 置換表エントリのフラグ */
const EXACT = 0;   // 正確なスコア
const LOWER = 1;   // 下限（α のカットオフ）
const UPPER = 2;   // 上限（β のカットオフ）

/** タイムチェック頻度（ノード数）。
 * per-node コストが最大 ~586µs のため 256 だと未チェック区間が ~150ms に達し、
 * 時間予算を大きく超過していた（例: 200ms 予算 → 実 640ms）。64 に短縮して
 * 予算遵守を改善する（Date.now() の追加コストは無視できる）。 */
const TIME_CHECK_INTERVAL = 64;

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
 * @param {(state, player) => number} evalFn - player 視点のスコアを返す評価関数
 * @returns {Array<{actions, board, summonCounts, _score}>}
 */
function sortCandidates(candidates, currentPlayer, boardSize, evalFn) {
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
      const score = evalFn(childState, currentPlayer);
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
 *   rng?: () => number,
 *   evalFn?: (state, player) => number
 * }} options
 *   evalFn 省略時は evaluate(state, player, weights) を使用する（既定動作は変更されない）。
 *   evalFn(state, player): player 視点のスコア（正=player有利）を返す非終端評価関数。
 *   NNUE 等の代替評価関数を注入できる。
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
    evalFn = (s, p) => evaluate(s, p, weights),
    // forward pruning 幅: 内部ノード(depth>=2)で分岐がこれを超えたら、
    // 1-ply 評価上位 forwardPruneWidth 手のみを再帰対象にする。
    // 0 以下で無効化（従来どおり全候補を探索＝厳密）。
    // 4×4 の典型分岐(~35)で発動しつつ盲点を抑えるため 24 を既定とする
    // （width掃引で 12〜24 が同一時間に平均+0.4手深く読めることを確認）。
    forwardPruneWidth = 24,
  } = options;

  const { currentPlayer, boardSize } = state;
  const maxSummons = maxSummonsFor(boardSize);
  const meta = { maxSummons, boardSize };

  const startTime = Date.now();
  let nodeCount = 0;
  let timedOut = false;

  // ── 探索用の可変盤面（1回だけ state.board をコピーし、以降は make/unmake で
  //    破壊的に更新する。フルクローンはこの1回のみ）。
  const searchBoard = state.board.map((row) => row.map((stack) => stack.slice()));
  const searchSummonCounts = { ...state.summonCounts };

  // Zobrist コンテキスト & 初期ハッシュ
  const zctx = createZobrist(boardSize);
  const rootHash = hashFromScratch(searchBoard, searchSummonCounts, currentPlayer, zctx);

  // 置換表（findBestTurn 呼び出しごとに新規作成）
  // キー: Zobrist ハッシュ（hi/lo ペア）を文字列化したもの + "|" + player
  // 値: { depth, score, flag, bestCandHash }
  const tt = new Map();

  // killer move ヒューリスティック: ply ごとに β カットを起こした手の署名を最大2つ保持。
  // 並べ替えで TT 最善手の次に優先し、α-β カットを増やす（厳密性は損なわない）。
  const killers = [];

  /** 候補手の軽量な署名（actions[0] のみ。盤面ハッシュ不要で安価）。 */
  function sigOf(cand) {
    const a = cand.actions[0];
    if (!a) return "";
    if (a.type === "move") {
      return "m" + a.from.r + "," + a.from.c + ">" + a.to.r + "," + a.to.c;
    }
    return a.type[0] + a.r + "," + a.c;
  }

  /** ply の killer スロットに署名を記録する（直近2手を保持）。 */
  function recordKiller(ply, sig) {
    let k = killers[ply];
    if (!k) { k = killers[ply] = [null, null]; }
    if (k[0] === sig) return;
    k[1] = k[0];
    k[0] = sig;
  }

  /**
   * タイムアウトチェック（nodeCount が CHECK_INTERVAL の倍数のときだけ実施）
   */
  function checkTime() {
    if (Date.now() - startTime >= timeBudgetMs) {
      timedOut = true;
    }
  }

  /**
   * candidate の署名として使う軽量キー（Zobrist ハッシュ文字列 + プレイヤー）。
   * candidate.board/summonCounts から from-scratch で計算する
   * （盤面は小さい (<=5x5) ため軽量。make/unmake の差分ハッシュは
   *  「現在の可変盤面上での実際の適用」にのみ使う）。
   */
  function candKey(cand, player) {
    return hashToKey(hashFromScratch(cand.board, cand.summonCounts, player, zctx)) + "|" + player;
  }

  /**
   * ネガマックス α-β 探索の本体（再帰）。
   * searchBoard / searchSummonCounts （可変・共有）を直接読み書きする。
   * 呼び出し前提: searchBoard/searchSummonCounts が「curState 相当」の状態になっていること。
   *
   * @param {string} player - 現在の手番（curState.currentPlayer 相当）
   * @param {{hi:number,lo:number}} curHash - 現在の局面の Zobrist ハッシュ
   * @param {number} depth  - 残り探索深さ（0 でリーフ評価）
   * @param {number} alpha
   * @param {number} beta
   * @param {number} ply    - ルートからの手数（勝ち/負けスコアの調整用）
   * @returns {number}  現在のプレイヤー視点のスコア
   */
  function negamax(player, curHash, depth, alpha, beta, ply) {
    nodeCount++;

    // タイムチェック（一定間隔）
    if (nodeCount % TIME_CHECK_INTERVAL === 0) {
      checkTime();
    }
    if (timedOut) return 0; // 中断: 0 を返す（この深さの結果は破棄される）

    const bs = boardSize;
    const curState = {
      board: searchBoard,
      summonCounts: searchSummonCounts,
      currentPlayer: player,
      boardSize: bs,
    };

    // ── 置換表参照 ──
    const posKey = hashToKey(curHash) + "|" + player;
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
      // リーフ評価は (局面, 手番) の純関数。同一局面が別経路や反復深化で
      // 再訪されることがあるため、結果を TT に depth=0 の EXACT として格納し
      // 再評価を省く（NNUE forward は重いため効果が大きい）。深さ >= 1 の
      // 探索は ttEntry.depth >= depth を満たさないので誤用されない。
      // リーフ由来のエントリ（depth===0）のみ再利用する。深い探索スコアを
      // リーフに混ぜると静的評価の意味が変わり棋力・決定性が変化するため、
      // depth>=1 のエントリは読まない・上書きしない。
      if (ttEntry && ttEntry.depth === 0) {
        return ttEntry.score;
      }
      const leafScore = evalFn(curState, player);
      if (!ttEntry) {
        tt.set(posKey, { depth: 0, score: leafScore, flag: EXACT, bestCandHash: null });
      }
      return leafScore;
    }

    // ── 手順並べ替え & forward pruning ──
    const opp = opponent(player);
    let orderedCandidates;

    // 内部ノード(depth>=2)で分岐が大きい場合は 1-ply 評価上位のみを再帰対象にする。
    const prune =
      forwardPruneWidth > 0 && depth >= 2 && candidates.length > forwardPruneWidth;

    if (prune) {
      // 1-ply 評価でスコアリング。同時に即勝ち(相手手なし)を検出したら確定して返す
      // （評価値だけで枝刈りすると詰みを見落とすため、ここで必ず全候補を確認する）。
      const scored = [];
      for (const c of candidates) {
        const childState = {
          board: c.board,
          summonCounts: c.summonCounts,
          currentPlayer: opp,
          boardSize: bs,
        };
        if (checkWinCondition(childState, meta)) {
          const score = WIN_SCORE - ply;
          const bch = candKey(c, opp);
          tt.set(posKey, { depth, score, flag: EXACT, bestCandHash: bch });
          return score;
        }
        scored.push({ c, s: evalFn(childState, player) });
      }
      scored.sort((a, b) => b.s - a.s);

      // TT 最善手は評価順位に関わらず必ず先頭に含める（証明済みの好手のため）。
      let head = null;
      if (ttBestCandHash !== null) {
        const hi = scored.findIndex((x) => candKey(x.c, opp) === ttBestCandHash);
        if (hi >= 0) {
          head = scored[hi].c;
          scored.splice(hi, 1);
        }
      }
      const keep = head ? forwardPruneWidth - 1 : forwardPruneWidth;
      const top = scored.slice(0, keep).map((x) => x.c);
      orderedCandidates = head ? [head, ...top] : top;
    } else {
      // 軽量並べ替え: TT最善手 → killer手 → エリミネート → 残り。
      const k = killers[ply];
      const ttFront = [];
      const killerArr = [];
      const elims = [];
      const rest = [];
      for (const c of candidates) {
        if (ttBestCandHash !== null && candKey(c, opp) === ttBestCandHash) {
          ttFront.push(c);
        } else if (k && (sigOf(c) === k[0] || sigOf(c) === k[1])) {
          killerArr.push(c);
        } else if (c.actions[0]?.type === "eliminate") {
          elims.push(c);
        } else {
          rest.push(c);
        }
      }
      orderedCandidates = [...ttFront, ...killerArr, ...elims, ...rest];
    }

    // ── 子ノード展開 ──
    let bestScore = -Infinity;
    let bestCandHash = null;
    const origAlpha = alpha;

    for (const cand of orderedCandidates) {
      if (timedOut) break;

      // ── make: actions を可変盤面に適用（破壊的）、ハッシュも差分更新 ──
      const { hash: childHash, undo } = applyTurn(
        searchBoard, searchSummonCounts, cand.actions, player, curHash, zctx
      );

      const childState = {
        board: searchBoard,
        summonCounts: searchSummonCounts,
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
        bestCandHash = hashToKey(childHash) + "|" + opp;
        undo(); // unmake
        // 置換表に記録して即リターン（これ以上良い手はない）
        tt.set(posKey, { depth, score, flag: EXACT, bestCandHash });
        return score;
      }

      // ── 再帰 ──
      const childScore = -negamax(opp, childHash, depth - 1, -beta, -alpha, ply + 1);

      undo(); // unmake: 探索前の状態に完全復帰

      if (timedOut) break;

      if (childScore > bestScore) {
        bestScore = childScore;
        bestCandHash = hashToKey(childHash) + "|" + opp;
      }
      if (childScore > alpha) alpha = childScore;
      if (alpha >= beta) {
        recordKiller(ply, sigOf(cand)); // β カットを起こした手を killer に記録
        break;
      }
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
    const sortedRoot = sortCandidates(rootCandidates, currentPlayer, boardSize, evalFn);

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

      // ── make: ルート候補を searchBoard に適用してから再帰、その後 unmake ──
      const { hash: childHash, undo } = applyTurn(
        searchBoard, searchSummonCounts, cand.actions, currentPlayer, rootHash, zctx
      );
      const score = -negamax(opp, childHash, depth - 1, -beta, -alpha, 1);
      undo();

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
