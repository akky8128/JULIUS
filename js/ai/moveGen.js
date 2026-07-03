/**
 * moveGen.js — 合法手全列挙モジュール（AI用）
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール。
 * simulateTurn を呼ばず、独自の軽量 apply ロジックで局面を差分更新する。
 */

import { normalizeStack, normalizeBoard, maxSummonsFor } from "../gameLogic.js";

// ───────────────────────── 定数 ─────────────────────────
/** 連続移動チェーンの最大手数 */
export const MAX_CHAIN_LENGTH = 8;
/** 1ターン生成の候補数上限（DFS 途中で超えたら打ち切り） */
export const MAX_CANDIDATES = 5000;

// ───────────────────────── ユーティリティ ─────────────────────────

/** プレイヤー文字を 1 文字に圧縮 */
const PC = { white: "w", black: "b" };

/**
 * 正規化済みボード（配列のみ）の文字列ハッシュを返す。
 * summonCounts も含めて一意性を確保する。
 * @param {Array[][]} board - 正規化済みボード（各セルは配列）
 * @param {{white:number,black:number}} summonCounts
 * @returns {string}
 */
export function hashPosition(board, summonCounts) {
  const rows = board.map(row =>
    row.map(stack => stack.map(p => PC[p] || p).join("")).join(",")
  ).join("|");
  return `${rows};${summonCounts.white}:${summonCounts.black}`;
}

/**
 * ボードをディープコピーする（各セルは配列のスライス）。
 * @param {Array[][]} board
 * @returns {Array[][]}
 */
function cloneBoard(board) {
  return board.map(row => row.map(stack => stack.slice()));
}

/** スタック先頭から player のコマが何枚連続しているか数える */
function ownTopCount(stack, player) {
  let count = 0;
  for (let i = stack.length - 1; i >= 0; i--) {
    if (stack[i] === player) count++;
    else break;
  }
  return count;
}

/** 盤内判定 */
function inBounds(r, c, boardSize) {
  return r >= 0 && r < boardSize && c >= 0 && c < boardSize;
}

/** 隣接4方向 */
const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];

// ───────────────────────── 軽量 apply ─────────────────────────

/**
 * ボードに移動を直接適用する（破壊的）。
 * @param {Array[][]} board
 * @param {{r:number,c:number}} from
 * @param {{r:number,c:number}} to
 * @param {number} count
 * @param {string} player
 * @returns {{ newOwnTop: number }} - 移動先での player コマの連続枚数
 */
function applyMoveInPlace(board, from, to, count, player) {
  const src = board[from.r][from.c];
  const moving = src.splice(src.length - count, count);
  const dst = board[to.r][to.c];
  dst.push(...moving);
  const newOwnTop = ownTopCount(dst, player);
  return { newOwnTop };
}

/**
 * ボードから移動を取り消す（破壊的）。
 * applyMoveInPlace と対になる。
 */
function undoMoveInPlace(board, from, to, count) {
  const dst = board[to.r][to.c];
  const moving = dst.splice(dst.length - count, count);
  const src = board[from.r][from.c];
  src.push(...moving);
}

// ───────────────────────── 主要エクスポート ─────────────────────────

/**
 * 指定状態から合法なターンをすべて列挙する。
 *
 * @param {{ board: Array[][], summonCounts: {white:number,black:number},
 *            currentPlayer: string, boardSize: number }} state
 *   board は正規化済み（各セルが配列）であること。呼び出し元で normalizeBoard を済ませること。
 * @returns {Array<{ actions: object[], board: Array[][], summonCounts: {white:number,black:number} }>}
 */
export function generateTurns(state) {
  const { board, summonCounts, currentPlayer, boardSize } = state;
  const maxSummons = maxSummonsFor(boardSize);
  const player = currentPlayer;
  const opponent = player === "white" ? "black" : "white";

  // 重複除去マップ: ハッシュ → { actions数, index }
  const seen = new Map();
  const results = [];
  let capped = false;

  /**
   * 候補を results に追加する（重複除去・effective pass 除去込み）。
   * @param {object[]} actions
   * @param {Array[][]} newBoard - 正規化済みのボード（コピー）
   * @param {{white:number,black:number}} newSummonCounts
   * @param {string} startHash - 開始局面のハッシュ（effective pass 検出用）
   */
  function addCandidate(actions, newBoard, newSummonCounts, startHash) {
    if (results.length >= MAX_CANDIDATES) {
      capped = true;
      return;
    }
    const hash = hashPosition(newBoard, newSummonCounts);
    // effective pass を除外（結果が開始局面と同一）
    if (hash === startHash) return;

    if (seen.has(hash)) {
      const existing = seen.get(hash);
      // より少ない手数のものを優先
      if (actions.length < existing.actionCount) {
        existing.actionCount = actions.length;
        results[existing.idx] = { actions: actions.slice(), board: newBoard, summonCounts: { ...newSummonCounts } };
      }
      return;
    }
    const idx = results.length;
    seen.set(hash, { actionCount: actions.length, idx });
    results.push({ actions: actions.slice(), board: newBoard, summonCounts: { ...newSummonCounts } });
  }

  const startHash = hashPosition(board, summonCounts);
  const summonPhaseOver = summonCounts.white === maxSummons && summonCounts.black === maxSummons;

  // ── 1. サモン ──────────────────────────────────────────────
  if (summonCounts[player] < maxSummons) {
    for (let r = 0; r < boardSize && !capped; r++) {
      for (let c = 0; c < boardSize && !capped; c++) {
        if (board[r][c].length === 0) {
          const newBoard = cloneBoard(board);
          newBoard[r][c] = [player];
          const newSC = { ...summonCounts, [player]: summonCounts[player] + 1 };
          addCandidate(
            [{ type: "summon", r, c }],
            newBoard,
            newSC,
            startHash
          );
        }
      }
    }
  }

  // ── 2. エリミネート ────────────────────────────────────────
  // 両プレイヤーがサモン完了後のみ
  if (summonPhaseOver) {
    for (let r = 0; r < boardSize && !capped; r++) {
      for (let c = 0; c < boardSize && !capped; c++) {
        const stack = board[r][c];
        if (stack.length === 0) continue;
        if (stack[stack.length - 1] !== player) continue;
        // 自分のコマの下に相手コマがあるか
        let opponentIndex = -1;
        for (let i = stack.length - 2; i >= 0; i--) {
          if (stack[i] !== player) {
            opponentIndex = i;
            break;
          }
        }
        if (opponentIndex === -1) continue;
        const newBoard = cloneBoard(board);
        newBoard[r][c].splice(opponentIndex, 1);
        addCandidate(
          [{ type: "eliminate", r, c }],
          newBoard,
          { ...summonCounts },
          startHash
        );
      }
    }
  }

  // ── 3. 移動チェーン（DFS） ─────────────────────────────────
  // 移動フェーズはサモン残りに関係なく常に試みる
  {
    const workingBoard = cloneBoard(board);
    const actionsStack = [];

    /**
     * DFS で移動チェーンを展開する。
     * @param {{r:number,c:number}|null} lastDest - 連続移動の直前の移動先（null なら1手目）
     * @param {number} depth - チェーンの深さ（0-indexed）
     */
    function dfsMove(lastDest, depth) {
      if (capped) return;

      // チェーン途中（depth > 0）でここまでの結果をすでに候補として登録
      if (depth > 0) {
        // ボードのスナップショットを取って候補登録
        const snapshot = cloneBoard(workingBoard);
        addCandidate(actionsStack, snapshot, { ...summonCounts }, startHash);
        if (capped) return;
      }

      if (depth >= MAX_CHAIN_LENGTH) return;

      // 移動元セルを決定（連続移動なら lastDest のみ）
      const sources = lastDest ? [lastDest] : null;

      const iterRows = sources
        ? sources
        : Array.from({ length: boardSize }, (_, r) => Array.from({ length: boardSize }, (_, c) => ({ r, c }))).flat();

      const cellsToTry = sources || iterRows;

      for (const fromCell of cellsToTry) {
        if (capped) return;
        const { r: fr, c: fc } = fromCell;
        const srcStack = workingBoard[fr][fc];
        const ownCount = ownTopCount(srcStack, player);
        if (ownCount === 0) continue;

        // 連続移動では1枚以上残す必要がある
        const maxMovable = depth > 0 ? ownCount - 1 : ownCount;
        if (maxMovable <= 0) continue;

        for (const [dr, dc] of DIRS) {
          if (capped) return;
          const tr = fr + dr;
          const tc = fc + dc;
          if (!inBounds(tr, tc, boardSize)) continue;

          for (let cnt = 1; cnt <= maxMovable; cnt++) {
            if (capped) return;
            const action = { type: "move", from: { r: fr, c: fc }, to: { r: tr, c: tc }, count: cnt };
            actionsStack.push(action);

            const { newOwnTop } = applyMoveInPlace(workingBoard, { r: fr, c: fc }, { r: tr, c: tc }, cnt, player);

            // 連続移動を続けるには移動先に自分のコマが2枚以上必要
            if (newOwnTop >= 2) {
              dfsMove({ r: tr, c: tc }, depth + 1);
            } else {
              // 連続移動できないが、現在の手数を候補として登録
              if (depth + 1 > 0) {
                const snapshot = cloneBoard(workingBoard);
                addCandidate(actionsStack, snapshot, { ...summonCounts }, startHash);
              }
            }

            undoMoveInPlace(workingBoard, { r: fr, c: fc }, { r: tr, c: tc }, cnt);
            actionsStack.pop();
          }
        }
      }
    }

    dfsMove(null, 0);
  }

  return results;
}

/**
 * 合法手が1つでも存在するか素早く確認する。
 * 完全列挙はせず、最初の合法手が見つかった時点で true を返す。
 *
 * @param {{ board: Array[][], summonCounts: {white:number,black:number},
 *            currentPlayer: string, boardSize: number }} state
 * @returns {boolean}
 */
export function hasAnyLegalTurn(state) {
  const { board, summonCounts, currentPlayer, boardSize } = state;
  const maxSummons = maxSummonsFor(boardSize);
  const player = currentPlayer;

  // サモン可能か
  if (summonCounts[player] < maxSummons) {
    for (let r = 0; r < boardSize; r++) {
      for (let c = 0; c < boardSize; c++) {
        if (board[r][c].length === 0) return true;
      }
    }
  }

  // エリミネート可能か（両者サモン完了後）
  const summonPhaseOver = summonCounts.white === maxSummons && summonCounts.black === maxSummons;
  if (summonPhaseOver) {
    for (let r = 0; r < boardSize; r++) {
      for (let c = 0; c < boardSize; c++) {
        const stack = board[r][c];
        if (stack.length === 0 || stack[stack.length - 1] !== player) continue;
        for (let i = stack.length - 2; i >= 0; i--) {
          if (stack[i] !== player) return true;
        }
      }
    }
  }

  // 移動可能か
  for (let r = 0; r < boardSize; r++) {
    for (let c = 0; c < boardSize; c++) {
      const stack = board[r][c];
      if (ownTopCount(stack, player) === 0) continue;
      for (const [dr, dc] of DIRS) {
        const tr = r + dr;
        const tc = c + dc;
        if (inBounds(tr, tc, boardSize)) return true;
      }
    }
  }

  return false;
}
