// ===================================================================
// このファイルは js/ai/nnue/network.js のミラーです。
// クライアント側（js/）とサーバー側（functions/）で同じロジックを共有するため、
// 両ファイルは常に同期を保つ必要があります。
// js/ai/nnue/network.js を変更した場合は、このファイルにも同じ変更を反映してください。
// ===================================================================

/**
 * network.js — NNUE方式評価関数のための JS 推論エンジン（Step 4）
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール。
 * Step 3 の PyTorch 学習パイプライン（tools/nnue/train.py）が出力する
 * models/gen001.json（メタ情報・テンソル形状）と models/gen001.bin
 * （float32 リトルエンディアンの重み列）を読み込み、順伝播を実行する。
 *
 * 数式は以下（すべて JS の Number = float64 で計算する。重みは float32 で
 * 格納されているが読み込み後は Number として扱われ float64 に広がるため、
 * tools/nnue/train_mock_test.py の reference_forward（float64 numpy 実装）
 * と一致する）:
 *
 *   x: 疎カウント入力（indices の各要素、重複込みで +1 相当）
 *   a1[o] = clippedReLU( b1[o] + Σ_{idx in indices} W1[o][idx] )   (o=0..H1-1)
 *   h1    = concat(a1, scalars)
 *   a2[o] = clippedReLU( b2[o] + Σ_j W2[o][j] * h1[j] )            (o=0..H2-1)
 *   a3[o] = clippedReLU( b3[o] + Σ_j W3[o][j] * a2[j] )            (o=0..H3-1)
 *   out   = b4[0] + Σ_j W4[0][j] * a3[j]                           (線形, スカラー)
 *   clippedReLU(v) = min(max(v, clipMin), clipMax)
 *
 * テンソルの形状・順序・オフセットは meta.tensors から動的に読み取り、
 * hidden サイズや featureDim が将来変わっても動作するようにする
 * （128 等の値をハードコードしない）。
 */

import { extractFeatures } from "./features.js";

// ───────────────────────── 内部ヘルパー ─────────────────────────

/**
 * meta.tensors を先頭から走査し、各テンソルの (name -> {offset, shape, size}) を求める。
 * float32 は1要素4バイトだが、ここでは Float32Array のインデックス単位（要素数）で
 * オフセットを計算する。
 * @param {object} meta - gen001.json をパースしたオブジェクト
 * @returns {{ offsets: Record<string, {offset:number, shape:number[], size:number}>, totalSize:number }}
 */
function computeTensorLayout(meta) {
  const offsets = {};
  let cursor = 0;
  for (const tensor of meta.tensors) {
    const size = tensor.shape.reduce((acc, d) => acc * d, 1);
    offsets[tensor.name] = { offset: cursor, shape: tensor.shape, size };
    cursor += size;
  }
  return { offsets, totalSize: cursor };
}

/**
 * clippedReLU: v を [clipMin, clipMax] にクリップする。
 * @param {number} v
 * @param {number} clipMin
 * @param {number} clipMax
 * @returns {number}
 */
function clippedReLU(v, clipMin, clipMax) {
  return Math.min(Math.max(v, clipMin), clipMax);
}

// ───────────────────────── 主要エクスポート ─────────────────────────

/**
 * 重み配列（Float32Array）とメタ情報から network オブジェクトを構築する。
 *
 * @param {Float32Array} weightsFloat32Array - gen001.bin をロードした Float32Array
 * @param {object} meta - gen001.json をパースしたオブジェクト
 *   (featureDim, scalarDim, hidden:[H1,H2,H3], clipMin, clipMax, tensors, ...)
 * @returns {{
 *   evaluate: (indices:number[], scalars:number[]) => number,
 *   evaluateSigmoid: (indices:number[], scalars:number[]) => number,
 *   evaluateState: (state:object, povPlayer?:string) => number,
 *   meta: object
 * }}
 */
export function createNetwork(weightsFloat32Array, meta) {
  const { offsets, totalSize } = computeTensorLayout(meta);

  if (weightsFloat32Array.length !== totalSize) {
    throw new Error(
      `createNetwork: weights length mismatch. expected ${totalSize} elements ` +
        `(derived from meta.tensors), got ${weightsFloat32Array.length}.`
    );
  }

  const featureDim = meta.featureDim;
  const scalarDim = meta.scalarDim;
  const [H1, H2, H3] = meta.hidden;
  const clipMin = meta.clipMin;
  const clipMax = meta.clipMax;

  const W1 = offsets.W1;
  const b1 = offsets.b1;
  const W2 = offsets.W2;
  const b2 = offsets.b2;
  const W3 = offsets.W3;
  const b3 = offsets.b3;
  const W4 = offsets.W4;
  const b4 = offsets.b4;

  // ── stack encoder（セル共有重みの per-cell エンコーダ）: meta.stackEncoder
  //    が存在する場合のみ有効化。存在しない旧モデルは従来の forward のまま。
  const hasStackEncoder = !!(meta.stackEncoder && offsets.Wenc && offsets.benc);
  const E = hasStackEncoder ? meta.stackEncoder.dim : 0;
  const NUM_CELLS = hasStackEncoder ? featureDim / 16 : 0; // 16 = MAX_DEPTH(8) * 2colors
  const CELL_SLOT_DIM = 16;
  const Wenc = hasStackEncoder ? offsets.Wenc : null;
  const benc = hasStackEncoder ? offsets.benc : null;
  const hasEncoderMaxPool = hasStackEncoder && !!meta.stackEncoder.maxPool;
  const encContrib = hasStackEncoder ? (hasEncoderMaxPool ? E * 2 : E) : 0;

  // ── scalars を a1(L1)経路にも注入: meta.scalarInL1 が true の場合のみ有効化。
  const hasScalarInL1 = !!(meta.scalarInL1 && offsets.Ws1);
  const Ws1 = hasScalarInL1 ? offsets.Ws1 : null;

  // ── 純エンコーダ版: meta.noFcPath が true の場合、従来の全結合L1(a1)経路を
  //    使わず、スタックエンコーダの pooled 出力のみで h1 を構成する。
  const hasNoFcPath = !!meta.noFcPath;
  const h1FcContrib = hasNoFcPath ? 0 : H1;

  const h1Len = h1FcContrib + encContrib + scalarDim;

  // 作業用バッファ（呼び出しごとの再確保を避ける）
  const a1 = hasNoFcPath ? null : new Float64Array(H1);
  const h1 = new Float64Array(h1Len);
  const a2 = new Float64Array(H2);
  const a3 = new Float64Array(H3);
  const perCell = hasStackEncoder
    ? new Float64Array(NUM_CELLS * CELL_SLOT_DIM)
    : null;
  const pooled = hasStackEncoder ? new Float64Array(E) : null;
  const pooledMax = hasEncoderMaxPool ? new Float64Array(E) : null;

  /**
   * 線形出力（活性化なしの生スコア）を計算する。
   * @param {number[]} indices - active な one-hot 特徴 index の配列（重複可）
   * @param {number[]} scalars - 長さ scalarDim のスカラー特徴
   * @returns {number}
   */
  function evaluate(indices, scalars) {
    if (!hasNoFcPath) {
      // L1: 疎 accumulator。o についてループし、各 active idx を加算する。
      const w1base = W1.offset;
      const ws1base = hasScalarInL1 ? Ws1.offset : 0;
      for (let o = 0; o < H1; o++) {
        let sum = weightsFloat32Array[b1.offset + o];
        const rowBase = w1base + o * featureDim;
        for (let k = 0; k < indices.length; k++) {
          sum += weightsFloat32Array[rowBase + indices[k]];
        }
        if (hasScalarInL1) {
          const ws1RowBase = ws1base + o * scalarDim;
          for (let s = 0; s < scalarDim; s++) {
            sum += weightsFloat32Array[ws1RowBase + s] * scalars[s];
          }
        }
        a1[o] = clippedReLU(sum, clipMin, clipMax);
      }

      // h1 = concat(a1, [pooled encoder出力], scalars)
      for (let o = 0; o < H1; o++) {
        h1[o] = a1[o];
      }
    }

    if (hasStackEncoder) {
      // perCell[cell][slot] を indices から再構築する（features.js の one-hot
      // レイアウトと同じ: idx = cell*16 + slot, slot = depth*2 + color）。
      perCell.fill(0);
      for (let k = 0; k < indices.length; k++) {
        const idx = indices[k];
        const cell = (idx / CELL_SLOT_DIM) | 0;
        const slot = idx % CELL_SLOT_DIM;
        perCell[cell * CELL_SLOT_DIM + slot] += 1;
      }

      // 共有重み Wenc[E,16] / benc[E] を全セルに適用し、sum-pool（＋max-pool）する。
      pooled.fill(0);
      if (hasEncoderMaxPool) pooledMax.fill(-Infinity);
      const wencBase = Wenc.offset;
      const bencBase = benc.offset;
      for (let cell = 0; cell < NUM_CELLS; cell++) {
        const cellBase = cell * CELL_SLOT_DIM;
        for (let e = 0; e < E; e++) {
          let sum = weightsFloat32Array[bencBase + e];
          const rowBase = wencBase + e * CELL_SLOT_DIM;
          for (let d = 0; d < CELL_SLOT_DIM; d++) {
            sum += weightsFloat32Array[rowBase + d] * perCell[cellBase + d];
          }
          const activated = clippedReLU(sum, clipMin, clipMax);
          pooled[e] += activated;
          if (hasEncoderMaxPool && activated > pooledMax[e]) pooledMax[e] = activated;
        }
      }

      for (let e = 0; e < E; e++) {
        h1[h1FcContrib + e] = pooled[e];
      }
      if (hasEncoderMaxPool) {
        for (let e = 0; e < E; e++) {
          h1[h1FcContrib + E + e] = pooledMax[e];
        }
      }
    }

    const encOffset = hasStackEncoder ? encContrib : 0;
    for (let s = 0; s < scalarDim; s++) {
      h1[h1FcContrib + encOffset + s] = scalars[s];
    }

    // L2: 密結合
    const w2base = W2.offset;
    for (let o = 0; o < H2; o++) {
      let sum = weightsFloat32Array[b2.offset + o];
      const rowBase = w2base + o * h1Len;
      for (let j = 0; j < h1Len; j++) {
        sum += weightsFloat32Array[rowBase + j] * h1[j];
      }
      a2[o] = clippedReLU(sum, clipMin, clipMax);
    }

    // L3: 密結合
    const w3base = W3.offset;
    for (let o = 0; o < H3; o++) {
      let sum = weightsFloat32Array[b3.offset + o];
      const rowBase = w3base + o * H2;
      for (let j = 0; j < H2; j++) {
        sum += weightsFloat32Array[rowBase + j] * a2[j];
      }
      a3[o] = clippedReLU(sum, clipMin, clipMax);
    }

    // L4: 線形出力層
    const w4base = W4.offset;
    let out = weightsFloat32Array[b4.offset];
    for (let j = 0; j < H3; j++) {
      out += weightsFloat32Array[w4base + j] * a3[j];
    }

    return out;
  }

  /**
   * evaluate の結果にシグモイドを適用した値を返す（[0,1] レンジ）。
   * @param {number[]} indices
   * @param {number[]} scalars
   * @returns {number}
   */
  function evaluateSigmoid(indices, scalars) {
    const out = evaluate(indices, scalars);
    return 1 / (1 + Math.exp(-out));
  }

  /**
   * state から特徴抽出して evaluate する便利関数。
   * @param {object} state - features.js の extractFeatures が期待する局面オブジェクト
   * @param {string} [povPlayer] - 省略時は state.currentPlayer
   * @returns {number}
   */
  function evaluateState(state, povPlayer) {
    const pov = povPlayer === undefined ? state.currentPlayer : povPlayer;
    const { indices, scalars } = extractFeatures(state, pov);
    return evaluate(indices, scalars);
  }

  return {
    evaluate,
    evaluateSigmoid,
    evaluateState,
    meta,
  };
}

/**
 * ArrayBuffer/Float32Array 形式の重みデータと meta から network を構築する
 * 低レベル API（createNetwork の薄いラッパー）。
 * @param {ArrayBuffer|Float32Array} arrayBufferOrFloat32
 * @param {object} meta
 * @returns {ReturnType<typeof createNetwork>}
 */
export function createNetworkFromBuffer(arrayBufferOrFloat32, meta) {
  const weights =
    arrayBufferOrFloat32 instanceof Float32Array
      ? arrayBufferOrFloat32
      : new Float32Array(arrayBufferOrFloat32);
  return createNetwork(weights, meta);
}

/**
 * json パス（または URL）から network をロードする。
 * 同じディレクトリ・同名の .bin ファイル（拡張子だけ .json → .bin に差し替え）を
 * 重みファイルとして読み込む。
 *
 * Node.js 環境: fs/promises を動的 import し、ローカルファイルとして読む。
 * ブラウザ環境: fetch でリクエストする。
 *
 * 環境判定は「window と fetch の両方がある」かどうかで行う。Node は該当しない
 * （file: パス文字列を fetch は扱えないため、ローカルパスが渡る Node では
 * 常に fs を優先する）。
 *
 * @param {string} jsonPathOrUrl - 例: "models/gen001.json" または "/models/gen001.json"
 * @returns {Promise<ReturnType<typeof createNetwork>>}
 */
export async function loadNetwork(jsonPathOrUrl) {
  const binPathOrUrl = jsonPathOrUrl.replace(/\.json$/, ".bin");
  const isBrowser =
    typeof window !== "undefined" && typeof fetch === "function";

  let meta;
  let weightsFloat32Array;

  if (isBrowser) {
    const [jsonRes, binRes] = await Promise.all([
      fetch(jsonPathOrUrl),
      fetch(binPathOrUrl),
    ]);
    meta = await jsonRes.json();
    const arrayBuffer = await binRes.arrayBuffer();
    weightsFloat32Array = new Float32Array(arrayBuffer);
  } else {
    const fs = await import("node:fs/promises");
    const jsonText = await fs.readFile(jsonPathOrUrl, "utf8");
    meta = JSON.parse(jsonText);
    const buf = await fs.readFile(binPathOrUrl);
    weightsFloat32Array = new Float32Array(
      buf.buffer,
      buf.byteOffset,
      buf.byteLength / 4
    );
  }

  return createNetwork(weightsFloat32Array, meta);
}
