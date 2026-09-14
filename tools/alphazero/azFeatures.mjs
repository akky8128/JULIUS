/**
 * azFeatures.mjs — AlphaZero移行 P1: ターン文脈ベクトル抽出＋D4アクション対称変換。
 *
 * ターン文脈ベクトル（TURN_CONTEXT_DIM = 26、PLAN.md §4.1）:
 *   [0..16]  チェーンヘッド位置 one-hot 17（16マス + [16]=「なし」）
 *            ヘッド = AUGMENT では A（積み増し元）、CHAIN では H（直前の移動先）。
 *            microActions.mjs の turnCtx.fromCell に一致する。
 *   [17..19] フェーズ one-hot 3（TURN_START / AUGMENT / CHAIN）
 *   [20..24] 積み増し方向 one-hot 5（4方向 + [24]=「なし」）
 *            方向インデックスは microActions.mjs / moveGen.js の DIRS 規約と同一:
 *            0=上(-1,0), 1=下(+1,0), 2=左(0,-1), 3=右(0,+1)
 *   [25]     ターン内マイクロ手数の正規化スカラー = min(microSteps, 16) / 16
 *            （16 は 4x4 の実用上限。超過分は 1.0 に飽和する）
 *
 * D4対称変換は js/ai/nnue/features.js の transformCell（sym 0..7）を唯一の情報源とし、
 * 方向の置換テーブルはそこから導出する（ハードコードしない）。
 */

import { transformCell, NUM_SYMMETRIES } from "../../js/ai/nnue/features.js";
import {
  NUM_MICRO_ACTIONS,
  END_TURN_ACTION,
  MOVE_ACTIONS_END,
  SUMMON_ACTIONS_START,
  SUMMON_ACTIONS_END,
  ELIMINATE_ACTIONS_START,
  ELIMINATE_ACTIONS_END,
} from "./microActions.mjs";

export const TURN_CONTEXT_DIM = 26;

/** ターン文脈ベクトル内オフセット（外部からの参照用） */
export const TC_HEAD_OFFSET = 0; // one-hot 17
export const TC_PHASE_OFFSET = 17; // one-hot 3
export const TC_DIR_OFFSET = 20; // one-hot 5
export const TC_STEPS_OFFSET = 25; // scalar 1

/** microSteps の正規化上限（min(microSteps,16)/16） */
export const MICRO_STEPS_NORM = 16;

const PHASE_INDEX = { TURN_START: 0, AUGMENT: 1, CHAIN: 2 };

/** DIRS 規約（microActions.mjs / moveGen.js と同一順序） */
const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];

// ───────────────────────── 方向置換テーブルの導出 ─────────────────────────

/**
 * transformCell は「回転/鏡映＋平行移動」のアフィン変換なので、方向ベクトルは
 * Δ = T(p + d) − T(p) で p に依存せず線形に写る。これを利用して sym ごとの
 * 方向置換 dirPerm[sym][dir] -> dir' を transformCell から導出する。
 */
function buildDirPermutations() {
  const N = 4; // 導出にのみ使用（Δは boardSize に依存しない）
  const perms = [];
  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    const base = transformCell(1, 1, sym, N);
    const perm = new Array(4);
    for (let dir = 0; dir < 4; dir++) {
      const [dr, dc] = DIRS[dir];
      const moved = transformCell(1 + dr, 1 + dc, sym, N);
      const ndr = moved.r - base.r;
      const ndc = moved.c - base.c;
      const target = DIRS.findIndex(([a, b]) => a === ndr && b === ndc);
      if (target === -1) throw new Error(`direction mapping failed for sym=${sym} dir=${dir}`);
      perm[dir] = target;
    }
    perms.push(perm);
  }
  return perms;
}

const DIR_PERMS = buildDirPermutations();

/** sym による方向インデックスの置換。 */
export function transformDir(dir, sym) {
  return DIR_PERMS[sym][dir];
}

// ───────────────────────── アクションindexのD4変換 ─────────────────────────

/**
 * マイクロアクションindexを D4 対称変換 sym で写す。
 * - 移動ノード cell*4+dir: セル置換＋方向置換
 * - 召喚/排除ノード 64+cell: セル置換
 * - 80（ターン終了）: 不動
 */
export function transformActionIndex(actionIdx, sym, boardSize = 4) {
  if (actionIdx === END_TURN_ACTION) return END_TURN_ACTION;
  if (actionIdx >= SUMMON_ACTIONS_START && actionIdx < SUMMON_ACTIONS_END) {
    const cell = actionIdx - SUMMON_ACTIONS_START;
    const r = Math.floor(cell / boardSize);
    const c = cell % boardSize;
    const t = transformCell(r, c, sym, boardSize);
    return SUMMON_ACTIONS_START + t.r * boardSize + t.c;
  }
  // 排除ノード（分離レイアウト時のみ出現）: セル置換は召喚と同一。
  if (actionIdx >= ELIMINATE_ACTIONS_START && actionIdx < ELIMINATE_ACTIONS_END) {
    const cell = actionIdx - ELIMINATE_ACTIONS_START;
    const r = Math.floor(cell / boardSize);
    const c = cell % boardSize;
    const t = transformCell(r, c, sym, boardSize);
    return ELIMINATE_ACTIONS_START + t.r * boardSize + t.c;
  }
  if (actionIdx >= 0 && actionIdx < MOVE_ACTIONS_END) {
    const cell = Math.floor(actionIdx / 4);
    const dir = actionIdx % 4;
    const r = Math.floor(cell / boardSize);
    const c = cell % boardSize;
    const t = transformCell(r, c, sym, boardSize);
    return (t.r * boardSize + t.c) * 4 + transformDir(dir, sym);
  }
  throw new Error(`invalid action index: ${actionIdx}`);
}

// ───────────────────────── ターン文脈ベクトル ─────────────────────────

/**
 * microActions.mjs の turnCtx からターン文脈ベクトル（Float64Array(26)）を抽出する。
 * TURN_START では head=なし([16])・dir=なし([24])・steps=0。
 */
export function extractTurnContext(turnCtx) {
  const vec = new Float64Array(TURN_CONTEXT_DIM);
  const boardSize = turnCtx.boardSize;

  if (turnCtx.phase === "TURN_START" || turnCtx.fromCell == null) {
    vec[TC_HEAD_OFFSET + 16] = 1;
  } else {
    vec[TC_HEAD_OFFSET + turnCtx.fromCell.r * boardSize + turnCtx.fromCell.c] = 1;
  }

  const phaseIdx = PHASE_INDEX[turnCtx.phase];
  if (phaseIdx === undefined) throw new Error(`unknown phase: ${turnCtx.phase}`);
  vec[TC_PHASE_OFFSET + phaseIdx] = 1;

  if (turnCtx.dir == null) {
    vec[TC_DIR_OFFSET + 4] = 1;
  } else {
    vec[TC_DIR_OFFSET + turnCtx.dir] = 1;
  }

  vec[TC_STEPS_OFFSET] = Math.min(turnCtx.microSteps ?? 0, MICRO_STEPS_NORM) / MICRO_STEPS_NORM;
  return vec;
}

/**
 * ターン文脈ベクトルを D4 対称変換 sym で写した新しいベクトルを返す。
 * ヘッドセル one-hot は transformCell、方向 one-hot は transformDir で置換。
 * フェーズ・「なし」スロット・手数スカラーは不変。
 */
export function transformTurnContext(vec, sym, boardSize = 4) {
  const out = new Float64Array(TURN_CONTEXT_DIM);

  // head one-hot 17
  for (let cell = 0; cell < boardSize * boardSize; cell++) {
    const v = vec[TC_HEAD_OFFSET + cell];
    if (v === 0) continue;
    const r = Math.floor(cell / boardSize);
    const c = cell % boardSize;
    const t = transformCell(r, c, sym, boardSize);
    out[TC_HEAD_OFFSET + t.r * boardSize + t.c] = v;
  }
  out[TC_HEAD_OFFSET + 16] = vec[TC_HEAD_OFFSET + 16];

  // phase one-hot 3（不変）
  for (let i = 0; i < 3; i++) out[TC_PHASE_OFFSET + i] = vec[TC_PHASE_OFFSET + i];

  // dir one-hot 5
  for (let dir = 0; dir < 4; dir++) {
    const v = vec[TC_DIR_OFFSET + dir];
    if (v !== 0) out[TC_DIR_OFFSET + transformDir(dir, sym)] = v;
  }
  out[TC_DIR_OFFSET + 4] = vec[TC_DIR_OFFSET + 4];

  out[TC_STEPS_OFFSET] = vec[TC_STEPS_OFFSET];
  return out;
}

export { NUM_MICRO_ACTIONS };
