/**
 * searchGen.js — 探索用の高速ムーブ適用/巻き戻し (make/unmake) ヘルパー
 *
 * generateTurns() (js/ai/moveGen.js, 変更禁止) が返す候補の `actions` を
 * 単一の可変ネスト配列ボードに対して破壊的に適用・巻き戻しする。
 *
 * 設計方針:
 *   - generateTurns() の結果集合（board/summonCounts の到達局面）は一切変更しない。
 *     search.js は引き続き generateTurns(state) を呼び、その結果 (candidate) を得る。
 *   - candidate.board は generateTurns 内部で cloneBoard() 済みの「新しい配列」。
 *     search.js 側ではこれをそのまま子ノードの board として使う代わりに、
 *     「actions を現在の可変ボードへ適用 → 探索 → 逆適用」という make/unmake 方式に
 *     切り替えることで、ノードごとのフルクローンを避ける。
 *   - actions の内容は summon 系(1件)/eliminate 系(1件)/move チェーン(複数件)のいずれかで、
 *     gameLogic.js の simulateTurn と同じ意味論（moveGen.js 内の apply ロジックと同一）。
 *   - Zobrist ハッシュは適用・巻き戻しと同時に差分更新する。
 *
 * 本モジュールは generateTurns の代替ではない。あくまで「候補の actions を
 * 可変ボードに再生する」ための補助であり、結果同一性は generateTurns 自体が担保する。
 */

import { xorPiece, xorSideToggle, xorSummon } from "./zobrist.js";

/** スタック先頭から player のコマが何枚連続しているか数える */
function ownTopCount(stack, player) {
  let count = 0;
  for (let i = stack.length - 1; i >= 0; i--) {
    if (stack[i] === player) count++;
    else break;
  }
  return count;
}

/**
 * 1ターン分の actions を可変ボードに適用する（破壊的）。
 * Zobrist ハッシュも同時に差分更新する。
 *
 * @param {Array[][]} board - 可変ボード（in-place 更新）
 * @param {{white:number,black:number}} summonCounts - 可変オブジェクト（in-place 更新）
 * @param {object[]} actions
 * @param {string} player - このターンの手番
 * @param {{hi:number,lo:number}} hash - 適用前のハッシュ
 * @param {object} zctx - createZobrist() の戻り値
 * @returns {{ hash: {hi:number,lo:number}, undo: () => void }}
 *   undo() を呼ぶと board/summonCounts/ハッシュが適用前の状態に完全復帰する。
 */
export function applyTurn(board, summonCounts, actions, player, hash, zctx) {
  const undoSteps = [];
  let h = hash;

  for (const action of actions) {
    if (action.type === "summon") {
      const { r, c } = action;
      const stack = board[r][c];
      const pos = stack.length; // 追加位置（下から数えて現在の長さ = 新しい位置）
      stack.push(player);
      h = xorPiece(h, zctx, r, c, pos, player);
      const old = summonCounts[player];
      summonCounts[player] = old + 1;
      h = xorSummon(h, zctx, player, old, old + 1);
      undoSteps.push(() => {
        stack.pop();
        summonCounts[player] = old;
      });
    } else if (action.type === "eliminate") {
      const { r, c } = action;
      const stack = board[r][c];
      let opponentIndex = -1;
      for (let i = stack.length - 2; i >= 0; i--) {
        if (stack[i] !== player) {
          opponentIndex = i;
          break;
        }
      }
      // 削除により opponentIndex 以降の全駒の「位置」が1つずつ繰り下がるため、
      // 旧配置(位置 i の駒)を全部XORで消し、splice後の新配置(位置 i の駒)を
      // 全部XORで入れ直す。これで正味「1枚減った」状態のハッシュになる。
      const oldTail = stack.slice(opponentIndex); // 旧: opponentIndex..end
      for (let i = 0; i < oldTail.length; i++) {
        h = xorPiece(h, zctx, r, c, opponentIndex + i, oldTail[i]); // 旧位置の駒を消す
      }
      const removed = stack.splice(opponentIndex, 1)[0];
      const newTail = stack.slice(opponentIndex); // 新: opponentIndex..end (1枚減)
      for (let i = 0; i < newTail.length; i++) {
        h = xorPiece(h, zctx, r, c, opponentIndex + i, newTail[i]); // 新位置の駒を入れる
      }
      undoSteps.push(() => {
        stack.splice(opponentIndex, 0, removed);
      });
    } else if (action.type === "move") {
      const { from, to, count } = action;
      const src = board[from.r][from.c];
      const dst = board[to.r][to.c];
      const srcLenBefore = src.length;
      const dstLenBefore = dst.length;
      const moving = src.splice(srcLenBefore - count, count);

      // ハッシュ: src 側の該当位置の駒を消す
      for (let i = 0; i < count; i++) {
        h = xorPiece(h, zctx, from.r, from.c, srcLenBefore - count + i, moving[i]);
      }
      dst.push(...moving);
      // ハッシュ: dst 側の新しい位置に駒を入れる
      for (let i = 0; i < count; i++) {
        h = xorPiece(h, zctx, to.r, to.c, dstLenBefore + i, moving[i]);
      }

      undoSteps.push(() => {
        const movedBack = dst.splice(dst.length - count, count);
        src.push(...movedBack);
      });
    } else {
      throw new Error(`searchGen: unknown action type: ${action.type}`);
    }
  }

  h = xorSideToggle(h, zctx);

  function undo() {
    for (let i = undoSteps.length - 1; i >= 0; i--) {
      undoSteps[i]();
    }
  }

  return { hash: h, undo };
}

export { ownTopCount };
