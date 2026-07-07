/**
 * zobrist.js — Zobrist ハッシュテーブル生成・差分更新プリミティブ
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール。
 *
 * ## 数値表現について
 * Zobrist ハッシュは XOR の組み合わせなので、32bit 整数の乱数を使い、
 * XOR 演算はすべて `>>> 0`（符号なし32bit）で行う。これにより:
 *   - JS の安全整数域 (2^53) を超えない（32bit の範囲に収まる）
 *   - ビット演算 `^` がそのまま使える（32bit 精度）
 *   - Map のキーとして使う際は `hash >>> 0` を文字列化するか、
 *     そのまま数値キーとして Map に渡せる
 * まれな衝突は avoid できないが、探索用 TT キーとしては
 * 32bit 単体では衝突確率がやや高いため、内部的には2つの32bit値
 * (hi, lo) を組み合わせた「64bit相当」の文字列キー
 * `hi.toString(36)+"_"+lo.toString(36)` を最終的な TT キーとして提供する
 * （xorAll 系関数は数値ペア {hi, lo} を返す設計）。
 *
 * ## 決定論的PRNG
 * mulberry32 を固定シードで使用し、同一実行間・環境間で常に同じテーブルを
 * 生成する（テストの再現性・キャッシュ可能性のため）。
 */

// ───────────────────────── PRNG ─────────────────────────

/** mulberry32: 決定論的 32bit PRNG */
function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0);
  };
}

const FIXED_SEED_A = 0x9e3779b9; // hi 系列用シード
const FIXED_SEED_B = 0x85ebca6b; // lo 系列用シード

// ───────────────────────── テーブル生成 ─────────────────────────

/**
 * boardSize に対応する Zobrist テーブルを生成する。
 *
 * テーブル構造:
 *   piece[r][c][stackPos][colorIdx] = {hi, lo}  … セル(r,c)のスタック内位置 stackPos に
 *                                                  色 colorIdx (0=white,1=black) の駒がある場合の値
 *   side = {hi, lo}                              … 手番が black のときに XOR する値
 *   summon[player][count] = {hi, lo}              … summonCounts[player]===count のときの値
 *                                                    (差分更新用に「countを1増減したときのXOR値」として扱う)
 *
 * スタック高さ上限は boardSize*boardSize（理論上の最大駒数）。
 * 位置は「下から数える」規約（index 0 = スタック最下段）。
 *
 * @param {number} boardSize
 * @returns {object} zobrist コンテキスト
 */
export function createZobrist(boardSize) {
  const rng = mulberry32(FIXED_SEED_A ^ (boardSize * 0x1000193));
  const rngLo = mulberry32(FIXED_SEED_B ^ (boardSize * 0x01000193));

  const maxStack = boardSize * boardSize;
  const maxSummons = Math.floor((boardSize * boardSize) / 2);

  function nextVal() {
    return { hi: rng(), lo: rngLo() };
  }

  // piece[r][c][pos][color]
  const piece = [];
  for (let r = 0; r < boardSize; r++) {
    const rowArr = [];
    for (let c = 0; c < boardSize; c++) {
      const posArr = [];
      for (let pos = 0; pos < maxStack; pos++) {
        posArr.push([nextVal(), nextVal()]); // [white, black]
      }
      rowArr.push(posArr);
    }
    piece.push(rowArr);
  }

  const side = nextVal();

  // summon[playerIdx][count] : XOR値そのもの（countごとに一意の値）。
  // 差分更新は summon[playerIdx][oldCount] を外し、summon[playerIdx][newCount] を入れる。
  const summon = [[], []]; // 0=white, 1=black
  for (let p = 0; p < 2; p++) {
    for (let n = 0; n <= maxSummons; n++) {
      summon[p].push(nextVal());
    }
  }

  return { boardSize, maxStack, maxSummons, piece, side, summon };
}

// ───────────────────────── 補助 ─────────────────────────

const COLOR_IDX = { white: 0, black: 1 };
const PLAYER_IDX = { white: 0, black: 1 };

/** {hi,lo} を XOR 合成する */
function xorInto(acc, val) {
  return { hi: (acc.hi ^ val.hi) >>> 0, lo: (acc.lo ^ val.lo) >>> 0 };
}

/** ハッシュペアを Map キーに使える文字列に変換する */
export function hashToKey(h) {
  return h.hi.toString(36) + "_" + h.lo.toString(36);
}

// ───────────────────────── from-scratch 計算 ─────────────────────────

/**
 * 盤面全体から Zobrist ハッシュをゼロから計算する。
 * @param {Array[][]} board - 正規化済みボード（各セルは配列、下から積み上げ）
 * @param {{white:number,black:number}} summonCounts
 * @param {string} currentPlayer
 * @param {object} ctx - createZobrist() の戻り値
 * @returns {{hi:number, lo:number}}
 */
export function hashFromScratch(board, summonCounts, currentPlayer, ctx) {
  let h = { hi: 0, lo: 0 };
  const { boardSize, piece } = ctx;
  for (let r = 0; r < boardSize; r++) {
    const row = board[r];
    if (!row) continue;
    for (let c = 0; c < boardSize; c++) {
      const stack = row[c];
      if (!stack || stack.length === 0) continue;
      for (let pos = 0; pos < stack.length; pos++) {
        const colorIdx = COLOR_IDX[stack[pos]];
        if (colorIdx === undefined) continue;
        h = xorInto(h, piece[r][c][pos][colorIdx]);
      }
    }
  }
  h = xorInto(h, ctx.summon[0][summonCounts.white]);
  h = xorInto(h, ctx.summon[1][summonCounts.black]);
  if (currentPlayer === "black") {
    h = xorInto(h, ctx.side);
  }
  return h;
}

// ───────────────────────── 差分更新プリミティブ ─────────────────────────

/**
 * セル(r,c)のスタック内位置posに色playerの駒を追加/削除するXOR値を返す
 * (XOR は自己逆演算なので追加・削除どちらも同じ関数でよい)。
 * @returns {{hi:number, lo:number}}
 */
export function pieceVal(ctx, r, c, pos, player) {
  const colorIdx = COLOR_IDX[player];
  return ctx.piece[r][c][pos][colorIdx];
}

/** 手番を反転させるXOR値を返す（xorInto(h, xorSide(ctx))） */
export function xorSide(ctx) {
  return ctx.side;
}

/**
 * 現在のハッシュに対して、セル(r,c)位置posの駒(player)をXORする。
 * @param {{hi:number,lo:number}} h
 * @returns {{hi:number,lo:number}} 新しいハッシュ
 */
export function xorPiece(h, ctx, r, c, pos, player) {
  return xorInto(h, pieceVal(ctx, r, c, pos, player));
}

/**
 * 手番をXORする（white<->black切り替え）。
 */
export function xorSideToggle(h, ctx) {
  return xorInto(h, ctx.side);
}

/**
 * summonCounts[player] を oldCount → newCount に変更した際のXOR差分を適用する。
 */
export function xorSummon(h, ctx, player, oldCount, newCount) {
  const p = PLAYER_IDX[player];
  let nh = xorInto(h, ctx.summon[p][oldCount]);
  nh = xorInto(nh, ctx.summon[p][newCount]);
  return nh;
}

export { xorInto };
