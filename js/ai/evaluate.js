/**
 * evaluate.js — 局面評価モジュール（AI用）
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール。
 * 評価値は正 = player 有利、負 = 相手有利。
 */

import { maxSummonsFor, normalizeStack } from "../gameLogic.js";

// ───────────────────────── 定数 ─────────────────────────

/**
 * 各評価項目の重み（デフォルト）。
 * 呼び出し元が weights を上書きすることで tuning できる。
 */
export const WEIGHTS = {
  top: 10,       // トップに立っているセル数（勝利条件に直結）
  buried: 3,     // 相手コマを埋めているセル数（行動不能にする）
  material: 4,   // 盤上 + 残サモン数の合計（駒の総量）
  elimChance: 2, // エリミネート可能なセル数（相手コマを除去できる機会）
  mobility: 1,   // 連続移動可能性のプロキシ（トップ連続枚数の合計）
};

/**
 * 終局を示すスコアの大きさ。
 * サーチで勝ちを見つけたときはこの値を使う。
 */
export const WIN_SCORE = 1e9;

// ───────────────────────── 内部ユーティリティ ─────────────────────────

/**
 * スタック先頭から player のコマが何枚連続しているか数える。
 * @param {string[]} stack
 * @param {string} player
 * @returns {number}
 */
function ownTopCount(stack, player) {
  let count = 0;
  for (let i = stack.length - 1; i >= 0; i--) {
    if (stack[i] === player) count++;
    else break;
  }
  return count;
}

// ───────────────────────── 主要エクスポート ─────────────────────────

/**
 * 局面を評価して player 視点のスコアを返す。
 * 終局判定は行わない（checkWinCondition は呼び出し元で行うこと）。
 *
 * シングルパスで全評価項目を計算するため GC 負荷が低い。
 *
 * @param {{ board: Array[][], summonCounts: {white:number,black:number},
 *            currentPlayer: string, boardSize: number }} state
 *   board は正規化済み（各セルが配列）であること。
 * @param {string} player - 評価の基準プレイヤー ("white" | "black")
 * @param {typeof WEIGHTS} weights
 * @returns {number}
 */
export function evaluate(state, player, weights = WEIGHTS) {
  const { board, summonCounts, boardSize } = state;
  const opponent = player === "white" ? "black" : "white";
  const maxSummons = maxSummonsFor(boardSize);

  // 各評価項目の my値 と opp値 を集計する
  let myTop = 0, oppTop = 0;
  let myBuried = 0, oppBuried = 0;     // myBuried = 自分のコマが上にいる相手コマ数
  let myElimChance = 0, oppElimChance = 0;
  let myMobility = 0, oppMobility = 0;

  for (let r = 0; r < boardSize; r++) {
    const row = board[r];
    if (!row) continue;
    for (let c = 0; c < boardSize; c++) {
      const stack = normalizeStack(row[c]);
      if (stack.length === 0) continue;

      const top = stack[stack.length - 1];

      // ── トップ ──
      if (top === player) {
        myTop++;

        // ── モビリティ（自分のコマが上に何枚連続か）──
        const ownCount = ownTopCount(stack, player);
        myMobility += ownCount;

        // ── 埋まっている相手コマ & エリミネートチャンス ──
        // スタック内に相手コマが1枚でもあれば +1
        let hasOppBelow = false;
        let buriedCount = 0;
        for (let i = stack.length - 2; i >= 0; i--) {
          if (stack[i] === opponent) {
            hasOppBelow = true;
            buriedCount++;
          }
        }
        myBuried += buriedCount;
        if (hasOppBelow) myElimChance++;

      } else {
        // top === opponent
        oppTop++;

        const ownCount = ownTopCount(stack, opponent);
        oppMobility += ownCount;

        let hasMyBelow = false;
        let buriedCount = 0;
        for (let i = stack.length - 2; i >= 0; i--) {
          if (stack[i] === player) {
            hasMyBelow = true;
            buriedCount++;
          }
        }
        oppBuried += buriedCount;
        if (hasMyBelow) oppElimChance++;
      }
    }
  }

  // ── マテリアル（盤上コマ + 残サモン数）──
  // 残サモン = maxSummons - summonCounts[p]
  // 盤上コマは top/buried から推定せず、
  // player の盤上コマ = myTop + (自分の埋まりコマ数) の代わりに
  // スタック全体を走査するのは重いので簡易版: top + buried で代替
  // （正確には全スタックを走査すべきだが、差分で使うので問題ない）
  const myMaterial = myTop + myBuried + (maxSummons - summonCounts[player]);
  const oppMaterial = oppTop + oppBuried + (maxSummons - summonCounts[opponent]);

  // ── 合計スコア（my - opp の差分にウェイトをかけて合計）──
  const score =
    weights.top        * (myTop        - oppTop       ) +
    weights.buried     * (myBuried     - oppBuried    ) +
    weights.material   * (myMaterial   - oppMaterial  ) +
    weights.elimChance * (myElimChance - oppElimChance) +
    weights.mobility   * (myMobility   - oppMobility  );

  return score;
}
