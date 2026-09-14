/**
 * features.js — NNUE方式評価関数のための特徴抽出モジュール（Step 1）
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール（Node専用APIは使用しない）。
 * ここで定義する仕様（one-hot index 式・スカラー特徴の定義・D4対称変換）は、
 * 今後の学習パイプライン（Python）と推論エンジン（JS）双方の契約となるため、
 * 変更する場合は両者を同時に更新すること。
 *
 * 評価値・POVの扱いは既存の evaluate.js の流儀を踏襲する:
 * 正 = povPlayer 有利、という向きで特徴を組み立てる（色は POV 相対でエンコードする）。
 */

import { maxSummonsFor, ruleMaxSummons, canEliminate, normalizeStack } from "../../gameLogic.js";

// ───────────────────────── 定数 ─────────────────────────

/**
 * スタック位置（下から数えた深さ）のクリップ上限。
 * 深さがこれ以上のコマは、すべて MAX_DEPTH-1 の位置に畳み込まれる。
 */
export const MAX_DEPTH = 8;

/**
 * スカラー特徴の個数。
 * [0] POV側の残サモン数(正規化), [1] 相手側の残サモン数(正規化),
 * [2] サモンフェーズ終了フラグ, [3] 空セル率,
 * [4] トップ数差(my-opp)/numCells, [5] 埋めコマ数差/numCells,
 * [6] エリミネート機会数差/numCells, [7] モビリティ差/numCells,
 * [8] マテリアル差/(2*maxSummons)
 *
 * [4]〜[8] は evaluate.js のヒューリスティック項を POV 相対の差分として供給する。
 * 盤面 one-hot からは間接的にしか学べない構造知識（トップ支配・埋め・除去機会・
 * 機動力・物量）を明示的に渡す狙い。
 * [9]〜[12] は勝敗規則（盤面充填時にトップ皆無の側が負け）に直結する終局近接シグナル
 * （絶対トップ密度・充填率との積）。積は線形スカラー層では表現できないため明示供給する。
 * いずれも盤面全体の集計量なので D4 対称変換で不変であり、augment 展開時に sym 間で
 * 使い回して問題ない。末尾追加なので旧モデル（scalarDim<13）は先頭のみ使用し無改変で動作。
 */
export const SCALAR_DIM = 13;

/**
 * D4群（正方形の対称変換）の要素数。
 * 恒等・90°/180°/270°回転・それぞれに鏡映を合成した4つ、の計8通り。
 */
export const NUM_SYMMETRIES = 8;

/**
 * 指定 boardSize での one-hot 特徴スロット数を返す。
 * 各セル × 深さ(MAX_DEPTH) × 色(2種) の組み合わせ数。
 * @param {number} boardSize
 * @returns {number}
 */
export function featureDim(boardSize) {
  return boardSize * boardSize * MAX_DEPTH * 2;
}

// ───────────────────────── 主要エクスポート ─────────────────────────

/**
 * 局面から NNUE 用の特徴を抽出する。
 *
 * one-hot 特徴（indices）:
 *   盤上の各コマについて、以下の式で算出した idx を列挙する。
 *     depthFromBottom = スタック配列中のインデックス i（0 が最下段）
 *     depthClipped    = Math.min(i, MAX_DEPTH - 1)
 *     color           = (コマの色 === povPlayer) ? 0 : 1   // POV相対
 *     idx = (r * boardSize + c) * MAX_DEPTH * 2 + depthClipped * 2 + color
 *   重複除去は行わない（クリップにより最上段付近で同一 idx が複数出現するのは
 *   「その位置に複数枚積まれている」ことを表す加算的表現として正しいため）。
 *   走査順は r → c → スタック下から上。
 *
 * スカラー特徴（scalars, 長さ SCALAR_DIM=4, 概ね [0,1] レンジ）:
 *   maxS = maxSummonsFor(boardSize)
 *   opp  = povPlayer の相手
 *   scalars[0] = (maxS - summonCounts[povPlayer]) / maxS   // POV側の残サモン数(正規化)
 *   scalars[1] = (maxS - summonCounts[opp])       / maxS   // 相手側の残サモン数(正規化)
 *   scalars[2] = (summonCounts.white === maxS && summonCounts.black === maxS) ? 1 : 0
 *   scalars[3] = emptyCellCount / (boardSize * boardSize)  // 空セル率
 *
 * @param {{ board: Array[][], summonCounts: {white:number,black:number},
 *            currentPlayer: string, boardSize: number }} state
 *   board は正規化済み（各セルが配列）であることを期待するが、防御的に
 *   normalizeStack を通す。
 * @param {string} povPlayer - 視点プレイヤー ("white" | "black")。
 *   通常は state.currentPlayer と同じだが、evaluate() と同様に明示引数として受け取る。
 * @returns {{ indices: number[], scalars: number[] }}
 */
export function extractFeatures(state, povPlayer) {
  const { board, summonCounts, boardSize } = state;
  const oppPlayer = povPlayer === "white" ? "black" : "white";
  const maxAll = ruleMaxSummons(boardSize);
  const myMax = maxAll[povPlayer];
  const oppMax = maxAll[oppPlayer];

  const indices = [];
  let emptyCellCount = 0;

  // ── ヒューリスティック集計量（POV 相対, evaluate.js と同じ定義）──
  let myTop = 0, oppTop = 0;
  let myBuried = 0, oppBuried = 0;        // 自トップ下に埋まっている相手コマ数（およびその逆）
  let myElim = 0, oppElim = 0;            // 相手コマを埋めている（=除去機会のある）セル数
  let myMobility = 0, oppMobility = 0;    // トップ連続同色枚数の合計

  for (let r = 0; r < boardSize; r++) {
    const row = board ? board[r] : undefined;
    for (let c = 0; c < boardSize; c++) {
      const stack = normalizeStack(row ? row[c] : undefined);
      if (stack.length === 0) {
        emptyCellCount++;
        continue;
      }
      const cellBase = (r * boardSize + c) * MAX_DEPTH * 2;
      for (let i = 0; i < stack.length; i++) {
        const piece = stack[i];
        const depthClipped = Math.min(i, MAX_DEPTH - 1);
        const color = piece === povPlayer ? 0 : 1;
        indices.push(cellBase + depthClipped * 2 + color);
      }

      // ── 集計（evaluate.js の単一パス集計を POV 相対で再現）──
      const top = stack[stack.length - 1];
      const owner = top === povPlayer ? "me" : "opp";
      // トップ連続同色枚数
      let run = 0;
      for (let i = stack.length - 1; i >= 0; i--) {
        if (stack[i] === top) run++; else break;
      }
      // トップ下に埋まっている「相手側（top から見た敵）」コマ数
      const foe = top === povPlayer ? oppPlayer : povPlayer;
      let buried = 0;
      for (let i = stack.length - 2; i >= 0; i--) {
        if (stack[i] === foe) buried++;
      }
      if (owner === "me") {
        myTop++; myMobility += run; myBuried += buried; if (buried > 0) myElim++;
      } else {
        oppTop++; oppMobility += run; oppBuried += buried; if (buried > 0) oppElim++;
      }
    }
  }

  const numCells = boardSize * boardSize;
  const myMaterial = myTop + myBuried + (myMax - summonCounts[povPlayer]);
  const oppMaterial = oppTop + oppBuried + (oppMax - summonCounts[oppPlayer]);

  const scalars = new Array(SCALAR_DIM);
  scalars[0] = (myMax - summonCounts[povPlayer]) / myMax;
  scalars[1] = (oppMax - summonCounts[oppPlayer]) / oppMax;
  scalars[2] = canEliminate(board, summonCounts, boardSize, povPlayer) ? 1 : 0;
  scalars[3] = emptyCellCount / numCells;
  scalars[4] = (myTop - oppTop) / numCells;
  scalars[5] = (myBuried - oppBuried) / numCells;
  scalars[6] = (myElim - oppElim) / numCells;
  scalars[7] = (myMobility - oppMobility) / numCells;
  scalars[8] = (myMaterial - oppMaterial) / (myMax + oppMax);
  // ── 終局近接シグナル（第2弾特徴, [9]〜[12]）──
  // 勝敗規則: 盤面が埋まった（or サモン枯渇）時点で「トップに自コマが1つも無い側が負け」。
  // よって「絶対トップ密度」と「盤面充填率との積」が終局勝敗に直結する。積は線形の
  // スカラー層(L2)では作れないため明示供給する。いずれも盤面集計量で D4 不変。
  const filled = (numCells - emptyCellCount) / numCells;
  scalars[9]  = myTop / numCells;                 // 自トップ密度（高いほど勝ちに近い）
  scalars[10] = oppTop / numCells;                // 相手トップ密度（→0 が勝ち条件）
  scalars[11] = (myTop / numCells) * filled;      // 充填×自支配（終局勝ち信号）
  scalars[12] = (oppTop / numCells) * filled;     // 充填×相手支配（終局負け信号）

  return { indices, scalars };
}

/**
 * 正方形の対称群 D4 における単一セル座標の変換を返す。
 * sym の意味（boardSize=N として、0-indexed座標 (r,c)）:
 *   0: 恒等                       (r, c)
 *   1: 反時計回り90°回転           (c, N-1-r)
 *   2: 180°回転                    (N-1-r, N-1-c)
 *   3: 反時計回り270°回転          (N-1-c, r)
 *   4: 水平鏡映（左右反転）         (r, N-1-c)
 *   5: 90°回転 + 鏡映（反対角線鏡映）(N-1-c, N-1-r)
 *   6: 180°回転 + 鏡映（垂直鏡映）   (N-1-r, c)
 *   7: 270°回転 + 鏡映（主対角線鏡映）(c, r)
 *
 * @param {number} r
 * @param {number} c
 * @param {number} sym - 0..7 の対称変換インデックス
 * @param {number} boardSize
 * @returns {{ r: number, c: number }}
 */
export function transformCell(r, c, sym, boardSize) {
  const N = boardSize;
  switch (sym) {
    case 0: return { r, c };
    case 1: return { r: c, c: N - 1 - r };
    case 2: return { r: N - 1 - r, c: N - 1 - c };
    case 3: return { r: N - 1 - c, c: r };
    case 4: return { r, c: N - 1 - c };
    case 5: return { r: N - 1 - c, c: N - 1 - r };
    case 6: return { r: N - 1 - r, c };
    case 7: return { r: c, c: r };
    default:
      throw new Error(`Invalid symmetry index: ${sym}`);
  }
}

/**
 * 局面を D4 対称変換した新しい state を返す（イミュータブル: 元の state は変更しない）。
 * board は各セル配列を深いコピーしつつ座標を transformCell で張り替える。
 * summonCounts / currentPlayer / boardSize はそのままコピーする。
 *
 * @param {{ board: Array[][], summonCounts: {white:number,black:number},
 *            currentPlayer: string, boardSize: number }} state
 * @param {number} sym - 0..7 の対称変換インデックス
 * @returns {{ board: Array[][], summonCounts: {white:number,black:number},
 *              currentPlayer: string, boardSize: number }}
 */
export function transformState(state, sym) {
  const { board, summonCounts, currentPlayer, boardSize } = state;

  const newBoard = Array.from({ length: boardSize }, () =>
    Array.from({ length: boardSize }, () => [])
  );

  for (let r = 0; r < boardSize; r++) {
    const row = board ? board[r] : undefined;
    for (let c = 0; c < boardSize; c++) {
      const stack = normalizeStack(row ? row[c] : undefined);
      const { r: nr, c: nc } = transformCell(r, c, sym, boardSize);
      newBoard[nr][nc] = stack.slice();
    }
  }

  return {
    board: newBoard,
    summonCounts: { ...summonCounts },
    currentPlayer,
    boardSize,
  };
}
