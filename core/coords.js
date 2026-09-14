// ===================================================================
// チュートリアル用の座標変換・棋譜記法ユーティリティ。
// DOM非依存の純関数のみを export し、Nodeからそのままテストできるようにする。
// 座標系・棋譜記法は game.html の実装（toCoord / diffToNotation）と厳密に一致させること。
// ===================================================================

/**
 * 棋譜座標文字列（例: "a1"）を盤面インデックス {r, c} に変換する。
 * game.html と同じ座標系: 列は 'a'始まり、行は上から数えて (boardSize - 段) 。
 * 例: boardSize=4 のとき a1 = {r:3, c:0}（左下）, a4 = {r:0, c:0}（左上）。
 */
export function parseCoord(str, boardSize) {
  if (typeof str !== "string" || str.length < 2) {
    throw new Error(`Invalid coordinate string: ${str}`);
  }
  const colChar = str[0];
  const rowPart = str.slice(1);
  const c = colChar.charCodeAt(0) - 97; // 'a' → 0
  const rank = Number(rowPart);
  if (!Number.isInteger(rank) || Number.isNaN(c)) {
    throw new Error(`Invalid coordinate string: ${str}`);
  }
  const r = boardSize - rank;
  if (r < 0 || r >= boardSize || c < 0 || c >= boardSize) {
    throw new Error(`Coordinate out of bounds: ${str}`);
  }
  return { r, c };
}

/**
 * 盤面インデックス {r, c} を棋譜座標文字列（例: "a1"）に変換する。
 */
export function formatCoord(r, c, boardSize) {
  return `${String.fromCharCode(97 + c)}${boardSize - r}`;
}

/**
 * 2つの盤面（denormalized: 空セル=0, 駒あり=配列）を比較し、
 * 「駒が減ったマス全て」→「駒が増えたマス全て」の順に座標トークンを連結した棋譜文字列を返す。
 * 同一マスで2枚以上変化した場合は座標の直前に枚数(係数)を付ける。
 * game.html の diffToNotation と同一ロジック。
 */
export function diffToNotation(boardBefore, boardAfter, boardSize) {
  const stackLength = (board, r, c) => {
    const cell = board?.[r]?.[c];
    return Array.isArray(cell) ? cell.length : 0;
  };

  const decreased = [];
  const increased = [];
  for (let r = 0; r < boardSize; r++) {
    for (let c = 0; c < boardSize; c++) {
      const delta = stackLength(boardAfter, r, c) - stackLength(boardBefore, r, c);
      if (delta < 0) decreased.push({ r, c, mag: -delta });
      else if (delta > 0) increased.push({ r, c, mag: delta });
    }
  }

  const toToken = ({ r, c, mag }) => (mag >= 2 ? mag : "") + formatCoord(r, c, boardSize);
  return [...decreased, ...increased].map(toToken).join("");
}
