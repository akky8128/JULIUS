/**
 * microActions.mjs — AlphaZero移行 P0: マイクロアクション81ノードのターン文法。
 *
 * ターンを「マイクロアクション」の列に分解し、固定81次元の行動空間を提供する。
 * 詳細仕様は tools/alphazero/PLAN.md §3 を参照。
 *
 * ノード規約:
 *   0..63  移動ノード   = cell*4 + dir  (cell = r*boardSize+c, dir: 0=上 1=下 2=左 3=右)
 *   64..79 召喚/排除ノード = 64 + cell   (召喚と排除は同時に合法にならないためノード共有)
 *   80     ターン終了ノード
 *
 * 正解集合(a)のリファレンスは js/ai/moveGen.js の generateTurns/hashPosition。
 * 本モジュールは既存コード（js/ 以下）を一切変更せず、読み取り専用で利用する。
 */

import { hashPosition, MAX_CHAIN_LENGTH } from "../../js/ai/moveGen.js";
import { normalizeBoard, maxSummonsFor, ruleMaxSummons, canEliminate, cloneBoard } from "../../js/gameLogic.js";

// 実験用: 排除をゲーム開始から解禁する場合、召喚と排除が同一TURN_STARTで同時に
// 選択肢になり得るため、共有ノード(64+cell)を分離する。環境変数 UL_SEP_ELIM で切替。
// 既定(未設定)は従来の81ノード共有レイアウトで、本番アプリ・既存ネットと完全一致。
//   分離時レイアウト(4x4フレーム=16セル基準):
//     移動 0..63 / 召喚 64..79 / 排除 80..95 / ターン終了 96  → NUM=97
export const SEPARATE_ELIMINATE =
  process.env.UL_SEP_ELIM === "1" || process.env.UL_SEP_ELIM === "true";

export const MOVE_ACTIONS_START = 0;
export const MOVE_ACTIONS_END = 64; // exclusive
export const SUMMON_ACTIONS_START = 64;
export const SUMMON_ACTIONS_END = 80; // exclusive
export const ELIMINATE_ACTIONS_START = 80; // 分離時のみ使用
export const ELIMINATE_ACTIONS_END = 96;   // exclusive（分離時のみ使用）
export const END_TURN_ACTION = SEPARATE_ELIMINATE ? 96 : 80;
export const NUM_MICRO_ACTIONS = SEPARATE_ELIMINATE ? 97 : 81;
export { MAX_CHAIN_LENGTH };

/** 移動方向: 0=上(-1,0) 1=下(+1,0) 2=左(0,-1) 3=右(0,+1)。moveGen.js の DIRS と同一順序。 */
const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];

// ───────────────────────── 座標/ノードindexユーティリティ ─────────────────────────

// 注: ownTopCount / inBounds は moveGen.js にも同等の非公開実装がある。
// 等価性テスト（microActions.test.mjs）は両実装の独立性を前提とするため、
// あえて共有せず本モジュール内に再実装している（意図的な重複）。
function ownTopCount(stack, player) {
  let count = 0;
  for (let i = stack.length - 1; i >= 0; i--) {
    if (stack[i] === player) count++;
    else break;
  }
  return count;
}

function inBounds(r, c, boardSize) {
  return r >= 0 && r < boardSize && c >= 0 && c < boardSize;
}

function cellIndex(r, c, boardSize) {
  return r * boardSize + c;
}

function cellFromIndex(idx, boardSize) {
  return { r: Math.floor(idx / boardSize), c: idx % boardSize };
}

/** 移動ノードindexを組み立てる。 */
export function moveNode(r, c, dir, boardSize) {
  return cellIndex(r, c, boardSize) * 4 + dir;
}

/** 召喚/排除ノードindexを組み立てる（非分離時は排除も同ノードを共有）。 */
export function summonNode(r, c, boardSize) {
  return SUMMON_ACTIONS_START + cellIndex(r, c, boardSize);
}

/** 排除ノードindexを組み立てる（分離時のみ使用）。 */
export function eliminateNode(r, c, boardSize) {
  return ELIMINATE_ACTIONS_START + cellIndex(r, c, boardSize);
}

export function decodeEliminateNode(idx, boardSize) {
  const cell = idx - ELIMINATE_ACTIONS_START;
  return cellFromIndex(cell, boardSize);
}

export function decodeMoveNode(idx, boardSize) {
  const cell = Math.floor(idx / 4);
  const dir = idx % 4;
  const { r, c } = cellFromIndex(cell, boardSize);
  return { r, c, dir };
}

export function decodeSummonNode(idx, boardSize) {
  const cell = idx - SUMMON_ACTIONS_START;
  return cellFromIndex(cell, boardSize);
}

/** スタックの中で自駒トップの直下にある、最上位の相手駒のインデックスを返す（無ければ-1）。 */
function findOpponentBelowTop(stack, player) {
  for (let i = stack.length - 2; i >= 0; i--) {
    if (stack[i] !== player) return i;
  }
  return -1;
}

/** ターン内反復禁止用のフルコンテキストキー（盤面ハッシュ＋フェーズ＋チェーンヘッド＋積み増し方向）。 */
function contextKey(board, summonCounts, phase, headCell, dir, boardSize) {
  const headIdx = headCell ? cellIndex(headCell.r, headCell.c, boardSize) : -1;
  return `${hashPosition(board, summonCounts)}|${phase}|${headIdx}|${dir ?? -1}`;
}

function isEffectivePass(ctx) {
  return hashPosition(ctx.board, ctx.summonCounts) === ctx.startHash;
}

/** board のクローン上で1枚だけ駒を動かす（自駒トップである前提はcaller側で検証済み）。 */
function boardAfterOneStep(board, from, to) {
  const next = cloneBoard(board);
  const src = next[from.r][from.c];
  const piece = src.pop();
  next[to.r][to.c].push(piece);
  return next;
}

// ───────────────────────── 初期化 ─────────────────────────

/**
 * ターン文脈オブジェクトを作る。
 *
 * 返り値の `done` フラグは「ターン完了」（召喚/排除/終了ノードの採用でこのターンの
 * マイクロアクション列が確定した）を意味し、「ゲーム終了」ではない。
 * 勝敗判定（相手に合法手があるか等）は呼び出し側の責務。
 *
 * @param {{board:Array, summonCounts:{white:number,black:number}, currentPlayer:string, boardSize:number}} state
 */
export function initTurnState(state) {
  const boardSize = state.boardSize;
  const board = cloneBoard(normalizeBoard(state.board, boardSize));
  const summonCounts = { ...state.summonCounts };
  const startHash = hashPosition(board, summonCounts);
  return {
    board,
    summonCounts,
    currentPlayer: state.currentPlayer,
    boardSize,
    phase: "TURN_START",
    fromCell: null,
    dir: null,
    destCell: null,
    depth: 0,
    microSteps: 0, // ターン内で採用したマイクロアクション数（azFeatures のターン文脈用）
    startHash,
    historyKeys: new Set(),
    done: false,
  };
}

// ───────────────────────── legalMicroActions ─────────────────────────

function legalAtTurnStart(ctx) {
  const { board, summonCounts, currentPlayer: player, boardSize } = ctx;
  const maxSummons = ruleMaxSummons(boardSize);
  const summonPhaseOver = canEliminate(board, summonCounts, boardSize, player);
  const legal = [];

  for (let r = 0; r < boardSize; r++) {
    for (let c = 0; c < boardSize; c++) {
      const stack = board[r][c];
      if (stack.length === 0) {
        if (summonCounts[player] < maxSummons[player]) legal.push(summonNode(r, c, boardSize));
      } else if (summonPhaseOver && stack[stack.length - 1] === player) {
        if (findOpponentBelowTop(stack, player) !== -1) {
          legal.push(SEPARATE_ELIMINATE ? eliminateNode(r, c, boardSize) : summonNode(r, c, boardSize));
        }
      }
      if (ownTopCount(stack, player) >= 1) {
        for (let dir = 0; dir < 4; dir++) {
          const [dr, dc] = DIRS[dir];
          if (inBounds(r + dr, c + dc, boardSize)) legal.push(moveNode(r, c, dir, boardSize));
        }
      }
    }
  }
  return legal;
}

/**
 * AUGMENT/CHAIN 共通の合法手計算。
 * @param {boolean} isInitialBlock - true なら AUGMENT（初手ブロック、残置制約なし）、false なら CHAIN（残置制約あり）
 */
function legalAtAugmentOrChain(ctx, isInitialBlock) {
  const { board, currentPlayer: player, boardSize, depth, historyKeys, fromCell, dir, destCell } = ctx;
  const legal = [];
  const nextPhase = isInitialBlock ? "AUGMENT" : "CHAIN";

  // 再採用（積み増し）: 残置制約なしなら own>=1、ありなら own>=2（1枚残す）
  const ownAtHead = ownTopCount(board[fromCell.r][fromCell.c], player);
  const minForReuse = isInitialBlock ? 1 : 2;
  if (ownAtHead >= minForReuse) {
    const reuseBoard = boardAfterOneStep(board, fromCell, destCell);
    const key = contextKey(reuseBoard, ctx.summonCounts, nextPhase, fromCell, dir, boardSize);
    if (!historyKeys.has(key)) legal.push(moveNode(fromCell.r, fromCell.c, dir, boardSize));
  }

  // 目的地からの連続移動（チェーン継続）
  const ownAtDest = ownTopCount(board[destCell.r][destCell.c], player);
  if (ownAtDest >= 2 && depth < MAX_CHAIN_LENGTH) {
    for (let dir2 = 0; dir2 < 4; dir2++) {
      const [dr, dc] = DIRS[dir2];
      const nr = destCell.r + dr;
      const nc = destCell.c + dc;
      if (!inBounds(nr, nc, boardSize)) continue;
      const chainBoard = boardAfterOneStep(board, destCell, { r: nr, c: nc });
      const key = contextKey(chainBoard, ctx.summonCounts, "CHAIN", destCell, dir2, boardSize);
      if (!historyKeys.has(key)) legal.push(moveNode(destCell.r, destCell.c, dir2, boardSize));
    }
  }

  if (!isEffectivePass(ctx)) legal.push(END_TURN_ACTION);
  return legal;
}

/**
 * ターン文脈から合法なマイクロアクションindexの配列を返す。
 *
 * 契約:
 * - `ctx.done === true`（ターン完了後）は常に空配列。
 * - `phase === "TURN_START"` で空配列 = 手番プレイヤーに合法手が一切ない
 *   （moveGen.js の hasAnyLegalTurn(state) === false と同値。このゲームでは敗北を意味する）。
 * - AUGMENT/CHAIN で空配列になるのは「実質パス盤面（終了ノード禁止）かつ反復禁止で
 *   全移動が刈られた」袋小路のみ。探索側はこの経路を単に放棄すればよい
 *   （フルコンテキストキーの性質上、到達可能な終端盤面は他経路で必ずカバーされる）。
 */
export function legalMicroActions(ctx) {
  if (ctx.done) return [];
  if (ctx.phase === "TURN_START") return legalAtTurnStart(ctx);
  if (ctx.phase === "AUGMENT") return legalAtAugmentOrChain(ctx, true);
  if (ctx.phase === "CHAIN") return legalAtAugmentOrChain(ctx, false);
  throw new Error(`unknown phase: ${ctx.phase}`);
}

/**
 * 合法手を Uint8Array(81) のマスクとして返すヘルパ（policy head 用）。
 * mask[i] === 1 ならノード i が合法。
 */
export function legalActionMask(ctx) {
  const mask = new Uint8Array(NUM_MICRO_ACTIONS);
  for (const idx of legalMicroActions(ctx)) mask[idx] = 1;
  return mask;
}

/**
 * 手番プレイヤーに合法なターンが1つでも存在するか（moveGen.js の hasAnyLegalTurn 相当）。
 * TURN_START で合法マイクロアクションが空 = 合法ターンなし（= 敗北）という契約の薄いラッパ。
 * @param {{board:Array, summonCounts:{white:number,black:number}, currentPlayer:string, boardSize:number}} state
 */
export function hasAnyLegalMicroTurn(state) {
  return legalMicroActions(initTurnState(state)).length > 0;
}

// ───────────────────────── applyMicroAction ─────────────────────────

function applyAtTurnStart(ctx, actionIdx) {
  const { board, summonCounts, currentPlayer: player, boardSize } = ctx;

  if (actionIdx >= SUMMON_ACTIONS_START && actionIdx < SUMMON_ACTIONS_END) {
    const { r, c } = decodeSummonNode(actionIdx, boardSize);
    const stack = board[r][c];
    const newBoard = cloneBoard(board);
    let newSummonCounts = summonCounts;
    if (stack.length === 0) {
      newBoard[r][c] = [player];
      newSummonCounts = { ...summonCounts, [player]: summonCounts[player] + 1 };
    } else {
      // 非分離レイアウトでの排除（同ノード共有）。
      const oppIdx = findOpponentBelowTop(stack, player);
      newBoard[r][c].splice(oppIdx, 1);
    }
    return { ...ctx, board: newBoard, summonCounts: newSummonCounts, done: true };
  }

  // 分離レイアウトでの排除ノード（80..95）。
  if (SEPARATE_ELIMINATE && actionIdx >= ELIMINATE_ACTIONS_START && actionIdx < ELIMINATE_ACTIONS_END) {
    const { r, c } = decodeEliminateNode(actionIdx, boardSize);
    const newBoard = cloneBoard(board);
    const oppIdx = findOpponentBelowTop(board[r][c], player);
    newBoard[r][c].splice(oppIdx, 1);
    return { ...ctx, board: newBoard, summonCounts, done: true };
  }

  if (actionIdx >= MOVE_ACTIONS_START && actionIdx < MOVE_ACTIONS_END) {
    const { r, c, dir } = decodeMoveNode(actionIdx, boardSize);
    const [dr, dc] = DIRS[dir];
    const from = { r, c };
    const to = { r: r + dr, c: c + dc };
    const newBoard = boardAfterOneStep(board, from, to);
    const key = contextKey(newBoard, summonCounts, "AUGMENT", from, dir, boardSize);
    return {
      ...ctx,
      board: newBoard,
      phase: "AUGMENT",
      fromCell: from,
      dir,
      destCell: to,
      depth: 1,
      microSteps: ctx.microSteps + 1,
      historyKeys: new Set(ctx.historyKeys).add(key),
    };
  }

  throw new Error(`illegal action ${actionIdx} at TURN_START`);
}

function applyAtAugmentOrChain(ctx, actionIdx) {
  const { board, summonCounts, boardSize, fromCell, dir, destCell, phase } = ctx;

  if (actionIdx === END_TURN_ACTION) {
    if (isEffectivePass(ctx)) throw new Error("illegal end-turn: effective pass");
    return { ...ctx, done: true };
  }

  if (actionIdx < MOVE_ACTIONS_START || actionIdx >= MOVE_ACTIONS_END) {
    throw new Error(`illegal action ${actionIdx} at ${phase}`);
  }

  const { r, c, dir: dir2 } = decodeMoveNode(actionIdx, boardSize);

  if (r === fromCell.r && c === fromCell.c && dir2 === dir) {
    // 再採用（積み増し）
    const newBoard = boardAfterOneStep(board, fromCell, destCell);
    const key = contextKey(newBoard, summonCounts, phase, fromCell, dir, boardSize);
    return {
      ...ctx,
      board: newBoard,
      microSteps: ctx.microSteps + 1,
      historyKeys: new Set(ctx.historyKeys).add(key),
    };
  }

  if (r === destCell.r && c === destCell.c) {
    // 連続移動（チェーン継続）
    const [dr, dc] = DIRS[dir2];
    const newDest = { r: destCell.r + dr, c: destCell.c + dc };
    const newBoard = boardAfterOneStep(board, destCell, newDest);
    const key = contextKey(newBoard, summonCounts, "CHAIN", destCell, dir2, boardSize);
    return {
      ...ctx,
      board: newBoard,
      phase: "CHAIN",
      fromCell: destCell,
      dir: dir2,
      destCell: newDest,
      depth: ctx.depth + 1,
      microSteps: ctx.microSteps + 1,
      historyKeys: new Set(ctx.historyKeys).add(key),
    };
  }

  throw new Error(`illegal action ${actionIdx} at ${phase}`);
}

/**
 * マイクロアクションを適用し、新しいターン文脈を返す（イミュータブル）。
 * ターン終了（召喚/排除/終了ノード採用）の場合は done:true を返す。
 */
export function applyMicroAction(ctx, actionIdx) {
  if (ctx.done) throw new Error("cannot apply action: turn already done");
  if (ctx.phase === "TURN_START") return applyAtTurnStart(ctx, actionIdx);
  if (ctx.phase === "AUGMENT" || ctx.phase === "CHAIN") return applyAtAugmentOrChain(ctx, actionIdx);
  throw new Error(`unknown phase: ${ctx.phase}`);
}
