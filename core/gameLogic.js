// Pure, DB-independent implementation of Ukeja Layer's turn rules.
// Ported from game.html's client logic (source of truth for the shipped
// ruleset) and cross-checked against the Python reference implementation
// (UkejaLayerGame.py). Operates on plain JS objects only so it can be
// unit-tested without touching Firebase.

export class GameLogicError extends Error {
  constructor(message, code = "failed-precondition") {
    super(message);
    this.name = "GameLogicError";
    this.code = code;
  }
}

export function maxSummonsFor(boardSize) {
  return Math.floor((boardSize * boardSize) / 2);
}

// ───────────────────────────────────────────────────────────────────────────
// 実験用ルール設定（環境変数駆動・既定は現行挙動を厳密に保存）。
//
// 3x3縮小盤での先手有利検証(AlphaZero流用)のために、非対称召喚数と排除解禁方式を
// 環境変数で切り替えられるようにする。環境変数が一切設定されていなければ、返り値は
// すべて従来の対称ルール（両者 maxSummonsFor / 両者召喚完了で排除解禁）と完全一致する
// ため、4x4本番アプリの挙動には一切影響しない。
//
//   UL_MAX_WHITE / UL_MAX_BLACK : 各色の召喚上限（未設定なら maxSummonsFor(boardSize)）
//   UL_ELIM_UNLOCK              : "both"(既定) 相手も含め全召喚完了で解禁
//                                 "self"       自分の全召喚完了で相手を待たず解禁
// ───────────────────────────────────────────────────────────────────────────

function ruleEnv(key) {
  return (typeof process !== "undefined" && process.env) ? process.env[key] : undefined;
}

/** 各色の召喚上限を返す。{white, black}。既定は対称 maxSummonsFor(boardSize)。 */
export function ruleMaxSummons(boardSize) {
  const def = maxSummonsFor(boardSize);
  const w = ruleEnv("UL_MAX_WHITE");
  const b = ruleEnv("UL_MAX_BLACK");
  return {
    white: w != null && w !== "" ? parseInt(w, 10) : def,
    black: b != null && b !== "" ? parseInt(b, 10) : def,
  };
}

/** 盤上に存在する player の駒数（スタック内の埋没駒も含む）。 */
function countOnBoard(board, boardSize, player) {
  let n = 0;
  for (let r = 0; r < boardSize; r++) {
    const row = board ? board[r] : undefined;
    for (let c = 0; c < boardSize; c++) {
      const stack = normalizeStack(row ? row[c] : undefined);
      for (let i = 0; i < stack.length; i++) if (stack[i] === player) n++;
    }
  }
  return n;
}

/**
 * player が排除を使えるか（排除解禁判定）。
 *  - "both"(既定): 両者が全召喚完了。従来の summonPhaseOver と同値。
 *  - "self": player 自身が「相手に排除された分を除く全ての駒」を召喚済み。
 *      eliminated = summonCounts[player] - 盤上player駒数（盤外に出るのは相手の排除のみ）。
 *      required   = maxSummons[player] - eliminated。
 *      解禁条件   = summonCounts[player] >= required。
 */
export function canEliminate(board, summonCounts, boardSize, player) {
  const max = ruleMaxSummons(boardSize);
  const mode = ruleEnv("UL_ELIM_UNLOCK") || "both";
  if (mode === "always") {
    // 排除をゲーム開始から解禁（召喚完了を待たない）。有効な排除対象があるかは
    // 呼び出し側（手生成）が別途チェックする。
    return true;
  }
  if (mode === "self") {
    const onBoard = countOnBoard(board, boardSize, player);
    const eliminated = summonCounts[player] - onBoard;
    const required = max[player] - eliminated;
    return summonCounts[player] >= required;
  }
  return summonCounts.white >= max.white && summonCounts.black >= max.black;
}

export function cloneBoard(board) {
  return board.map((row) => row.map((stack) => normalizeStack(stack).slice()));
}

// Board cells may be `0`, `null`, or an array in the stored format; treat
// anything that isn't a non-empty array as an empty stack.
export function normalizeStack(cell) {
  return Array.isArray(cell) ? cell : [];
}

// Realtime Database omits empty arrays/rows, so a stored board can arrive
// as a sparse or short-rowed structure. Rebuild a full boardSize x boardSize
// grid before running any logic against it.
export function normalizeBoard(board, boardSize) {
  const rows = [];
  for (let r = 0; r < boardSize; r++) {
    const sourceRow = board ? board[r] : undefined;
    const row = [];
    for (let c = 0; c < boardSize; c++) {
      row.push(normalizeStack(sourceRow ? sourceRow[c] : undefined));
    }
    rows.push(row);
  }
  return rows;
}

// Mirrors game.html's per-turn timer deduction (delay-then-countdown),
// but uses the trusted server clock (`nowMs`) instead of the client's.
export function computeTimersAfterTurn(lastMove, timeControl, moverColor, nowMs) {
  const newTimers = {...lastMove.timers};
  if (!timeControl || !timeControl.enabled) return newTimers;
  if (typeof lastMove.timestamp !== "number") return newTimers;

  const remaining = lastMove.timers[moverColor];
  const elapsedMs = Math.max(0, nowMs - lastMove.timestamp);
  const elapsedSec = Math.floor(elapsedMs / 1000);
  const timeToDeduct = Math.max(0, elapsedSec - timeControl.delay);
  newTimers[moverColor] = Math.max(0, remaining - timeToDeduct);
  return newTimers;
}

function inBounds(r, c, boardSize) {
  return r >= 0 && r < boardSize && c >= 0 && c < boardSize;
}

function isAdjacent(fromR, fromC, toR, toC) {
  const dr = Math.abs(fromR - toR);
  const dc = Math.abs(fromC - toC);
  return (dr === 1 && dc === 0) || (dr === 0 && dc === 1);
}

// Number of the mover's own pieces stacked contiguously at the top.
function ownTopCount(stack, player) {
  let count = 0;
  for (let i = stack.length - 1; i >= 0; i--) {
    if (stack[i] === player) count++;
    else break;
  }
  return count;
}

function applySummon(board, summonCounts, action, player, boardSize, maxSummons) {
  const {r, c} = action;
  if (!inBounds(r, c, boardSize)) {
    throw new GameLogicError("Summon target is out of bounds.", "invalid-argument");
  }
  const stack = normalizeStack(board[r][c]);
  if (stack.length > 0) {
    throw new GameLogicError("Summon target is not empty.", "failed-precondition");
  }
  if (summonCounts[player] >= maxSummons) {
    throw new GameLogicError("No summons remaining.", "failed-precondition");
  }
  board[r][c] = [...stack, player];
  summonCounts[player] += 1;
}

function applyEliminate(board, summonCounts, action, player, boardSize, maxSummons) {
  const {r, c} = action;
  if (!inBounds(r, c, boardSize)) {
    throw new GameLogicError("Eliminate target is out of bounds.", "invalid-argument");
  }
  const summonPhaseOver = summonCounts.white === maxSummons && summonCounts.black === maxSummons;
  if (!summonPhaseOver) {
    throw new GameLogicError("Cannot eliminate before both players finish summoning.", "failed-precondition");
  }
  const stack = normalizeStack(board[r][c]).slice();
  if (stack.length === 0 || stack[stack.length - 1] !== player) {
    throw new GameLogicError("Your own piece must be on top of the chosen cell.", "failed-precondition");
  }
  let opponentIndex = -1;
  for (let i = stack.length - 2; i >= 0; i--) {
    if (stack[i] !== player) {
      opponentIndex = i;
      break;
    }
  }
  if (opponentIndex === -1) {
    throw new GameLogicError("There is no opponent piece to eliminate at that cell.", "failed-precondition");
  }
  stack.splice(opponentIndex, 1);
  board[r][c] = stack;
}

function applyMove(board, action, player, boardSize, continuous) {
  const {from, to, count} = action;
  if (!from || !to || !Number.isInteger(count) || count < 1) {
    throw new GameLogicError("Malformed move action.", "invalid-argument");
  }
  if (!inBounds(from.r, from.c, boardSize) || !inBounds(to.r, to.c, boardSize)) {
    throw new GameLogicError("Move references a cell outside the board.", "invalid-argument");
  }
  if (!isAdjacent(from.r, from.c, to.r, to.c)) {
    throw new GameLogicError("Move destination must be orthogonally adjacent.", "invalid-argument");
  }
  if (continuous.active) {
    if (from.r !== continuous.lastDest.r || from.c !== continuous.lastDest.c) {
      throw new GameLogicError("Continuous move must start from the previous destination.", "failed-precondition");
    }
  }

  const sourceStack = normalizeStack(board[from.r][from.c]).slice();
  const ownCount = ownTopCount(sourceStack, player);

  if (ownCount === 0) {
    throw new GameLogicError("You have no piece on top of the source cell.", "failed-precondition");
  }
  const maxMovable = continuous.active ? ownCount - 1 : ownCount;
  if (maxMovable <= 0) {
    throw new GameLogicError("A continuous move must leave one of your own pieces behind.", "failed-precondition");
  }
  if (count > maxMovable) {
    throw new GameLogicError("Cannot move that many pieces from the source cell.", "failed-precondition");
  }

  const movingPieces = sourceStack.slice(sourceStack.length - count);
  const remainingSource = sourceStack.slice(0, sourceStack.length - count);
  board[from.r][from.c] = remainingSource;

  const destStack = normalizeStack(board[to.r][to.c]).slice();
  board[to.r][to.c] = [...destStack, ...movingPieces];

  const newOwnTop = ownTopCount(board[to.r][to.c], player);
  return {
    lastDest: {r: to.r, c: to.c},
    canContinue: newOwnTop > 1,
  };
}

// Mirrors game.html's checkWinCondition exactly: `state` is the state that
// would become current *after* the turn (i.e. currentPlayer is the player
// about to move next). Returns true if that player has no legal move.
export function checkWinCondition(state, meta) {
  const {board, summonCounts, currentPlayer} = state;
  const {maxSummons, boardSize} = meta;
  const opponent = currentPlayer;

  if (summonCounts[opponent] < maxSummons) {
    for (let r = 0; r < boardSize; r++) {
      if (!board[r]) return false;
      for (let c = 0; c < boardSize; c++) {
        if (normalizeStack(board[r][c]).length === 0) return false;
      }
    }
  }

  for (let r = 0; r < boardSize; r++) {
    if (!board[r]) continue;
    for (let c = 0; c < boardSize; c++) {
      const stack = normalizeStack(board[r][c]);
      if (stack.length > 0 && stack[stack.length - 1] === opponent) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Replays a full turn's worth of actions against a trusted starting state.
 *
 * A turn is either:
 *  - exactly one "summon" action, or
 *  - exactly one "eliminate" action, or
 *  - one or more "move" actions chained as continuous moves.
 *
 * Throws GameLogicError if any action is illegal, if the action types are
 * mixed, or if the net effect of the whole turn leaves the board unchanged
 * (the "effective pass" is forbidden by the rulebook).
 */
export function simulateTurn({board, summonCounts, currentPlayer, boardSize}, actions) {
  if (!Array.isArray(actions) || actions.length === 0) {
    throw new GameLogicError("At least one action is required.", "invalid-argument");
  }

  const maxSummons = maxSummonsFor(boardSize);
  const startBoard = cloneBoard(board);
  const workingBoard = cloneBoard(board);
  const workingSummonCounts = {...summonCounts};
  const player = currentPlayer;

  const continuous = {active: false, lastDest: null};
  let turnKind = null;

  for (const action of actions) {
    if (!action || typeof action.type !== "string") {
      throw new GameLogicError("Malformed action.", "invalid-argument");
    }

    if (action.type === "summon") {
      if (turnKind !== null) {
        throw new GameLogicError("Summon must be the only action in a turn.", "failed-precondition");
      }
      applySummon(workingBoard, workingSummonCounts, action, player, boardSize, maxSummons);
      turnKind = "summon";
    } else if (action.type === "eliminate") {
      if (turnKind !== null) {
        throw new GameLogicError("Eliminate must be the only action in a turn.", "failed-precondition");
      }
      applyEliminate(workingBoard, workingSummonCounts, action, player, boardSize, maxSummons);
      turnKind = "eliminate";
    } else if (action.type === "move") {
      if (turnKind !== null && turnKind !== "move") {
        throw new GameLogicError("Move cannot be combined with other action types.", "failed-precondition");
      }
      const result = applyMove(workingBoard, action, player, boardSize, continuous);
      turnKind = "move";
      continuous.active = true;
      continuous.lastDest = result.lastDest;
    } else {
      throw new GameLogicError(`Unknown action type: ${action.type}`, "invalid-argument");
    }
  }

  if (JSON.stringify(startBoard) === JSON.stringify(workingBoard)) {
    throw new GameLogicError("A turn must change the board (no effective pass).", "failed-precondition");
  }

  return {
    board: denormalizeBoard(workingBoard),
    summonCounts: workingSummonCounts,
  };
}

// Realtime Database treats a written empty array/object as "no value" and
// drops that node entirely (and, transitively, any row that ends up with no
// surviving cells). Storing `0` for empty cells instead of `[]` keeps every
// row/column present after a write, mirroring the format createGame uses
// for a fresh board. normalizeStack() already treats `0` as an empty stack.
export function denormalizeBoard(board) {
  return board.map((row) => row.map((stack) => (
    Array.isArray(stack) && stack.length === 0 ? 0 : stack
  )));
}
