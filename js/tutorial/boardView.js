// ===================================================================
// チュートリアル用の盤面レンダラ（DOM依存）。
// game.html の renderBoard(line 1059〜) を簡略移植したもの。
// FLIPアニメーションは Phase 1 では不要なのでテレポート描画のみ行う。
// ===================================================================

import { formatCoord } from "./coords.js";

/**
 * 盤面を containerEl に描画する。
 *
 * @param {HTMLElement} containerEl - 盤面を描画するコンテナ要素（外周に座標ラベルを重ねるため
 *   position 指定可能な要素であることが望ましい。内部で position:relative を強制しない）。
 * @param {Array<Array<0|string[]>>} boardData - denormalized board（空セル=0, 駒あり=配列）。
 * @param {number} boardSize
 * @param {object} [options]
 * @param {{r:number,c:number}|null} [options.selected] - 選択中のマス。
 * @param {Array<{r:number,c:number}>} [options.validMoves] - 移動可能マスの一覧。
 * @param {string[]} [options.highlights] - 光らせるマスの棋譜座標文字列一覧（例 ['c3']）。
 * @param {(r:number, c:number) => void} [options.onCellClick] - セルクリック時のコールバック。
 */
export function renderBoard(containerEl, boardData, boardSize, options = {}) {
  const { selected = null, validMoves = [], highlights = [], onCellClick = null } = options;

  containerEl.innerHTML = "";
  containerEl.style.display = "grid";
  containerEl.style.gridTemplateColumns = `repeat(${boardSize}, 1fr)`;
  containerEl.style.gridTemplateRows = `repeat(${boardSize}, 1fr)`;

  const highlightSet = new Set(highlights || []);

  for (let r = 0; r < boardSize; r++) {
    for (let c = 0; c < boardSize; c++) {
      const cell = document.createElement("div");
      cell.dataset.r = String(r);
      cell.dataset.c = String(c);
      cell.className = `cell ${(r + c) % 2 === 0 ? "light" : "dark"}`;

      if (selected && selected.r === r && selected.c === c) {
        cell.classList.add("selected");
      }
      if (validMoves.some((m) => m.r === r && m.c === c)) {
        cell.classList.add("valid-move");
      }
      if (highlightSet.has(formatCoord(r, c, boardSize))) {
        cell.classList.add("tutorial-highlight");
      }

      // 外周に座標ラベルを常時表示する（チュートリアル向け）。
      appendCoordLabels(cell, r, c, boardSize);

      const stack = Array.isArray(boardData?.[r]?.[c]) ? boardData[r][c] : [];
      stack.forEach((pieceColor, i) => {
        const piece = document.createElement("div");
        piece.className = `piece ${pieceColor}`;
        piece.style.transform = `translateY(${-i * 8}px)`;
        piece.style.zIndex = String(i + 1);
        cell.appendChild(piece);
      });

      if (typeof onCellClick === "function") {
        cell.addEventListener("click", () => onCellClick(r, c));
      }

      containerEl.appendChild(cell);
    }
  }
}

// 盤面の一番外側の行・列にだけ、隅に小さく座標ラベル（列= a〜, 行=1〜）を重ねる。
function appendCoordLabels(cell, r, c, boardSize) {
  const labels = [];
  if (r === boardSize - 1) {
    // 一番下の行にだけ列ラベル(a,b,c...)を表示
    labels.push({ text: String.fromCharCode(97 + c), cls: "tutorial-coord-col" });
  }
  if (c === 0) {
    // 一番左の列にだけ行ラベル(1,2,3...)を表示
    labels.push({ text: String(boardSize - r), cls: "tutorial-coord-row" });
  }
  labels.forEach(({ text, cls }) => {
    const label = document.createElement("span");
    label.className = `tutorial-coord-label ${cls}`;
    label.textContent = text;
    cell.appendChild(label);
  });
}
