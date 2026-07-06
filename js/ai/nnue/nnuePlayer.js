/**
 * nnuePlayer.js — NNUE 評価関数を使う探索プレイヤー（Step 5）
 *
 * ブラウザ・Node.js 両対応の純粋 ESM モジュール。
 * js/ai/search.js の findBestTurn に evalFn として net.evaluateState を注入する。
 */

import { loadNetwork } from "./network.js";
import { findBestTurn } from "../search.js";

/**
 * netPath（.json または .bin）を .json パスに正規化する。
 * @param {string} netPath
 * @returns {string}
 */
function normalizeNetPath(netPath) {
  return netPath.replace(/\.bin$/, ".json");
}

/**
 * ファイルパスからベース名（拡張子なし）を取り出す（表示名用）。
 * @param {string} p
 * @returns {string}
 */
function basename(p) {
  const parts = p.replace(/\\/g, "/").split("/");
  const last = parts[parts.length - 1] || p;
  return last.replace(/\.json$/, "");
}

/**
 * 既にロード済みの network から同期的に NNUE 探索プレイヤーを生成する。
 *
 * @param {() => number} rng - [0,1) の乱数を返す関数（タイブレーク用）
 * @param {ReturnType<typeof import("./network.js").createNetwork>} net
 * @param {{
 *   maxDepth?: number,
 *   timeBudgetMs?: number,
 *   name?: string
 * }} [options]
 * @returns {{ name: string, chooseTurn(state): object|null,
 *             lastSearchInfo: {depthReached:number, nodes:number, elapsedMs:number}|null }}
 */
export function createNnueSearchPlayerFromNet(rng, net, options = {}) {
  const { maxDepth = 64, timeBudgetMs = 1000, name } = options;

  let lastSearchInfo = null;

  return {
    name: name || `nnue(net,d${maxDepth},${timeBudgetMs}ms)`,
    get lastSearchInfo() { return lastSearchInfo; },
    chooseTurn(state) {
      const result = findBestTurn(state, {
        maxDepth,
        timeBudgetMs,
        rng,
        evalFn: (s, p) => net.evaluateState(s, p),
      });
      lastSearchInfo = {
        depthReached: result.depthReached,
        nodes:        result.nodes,
        elapsedMs:    result.elapsedMs,
      };
      return result.turn;
    },
  };
}

/**
 * netPath から network をロードし、NNUE 探索プレイヤーを生成する（非同期）。
 *
 * @param {() => number} rng - [0,1) の乱数を返す関数
 * @param {{
 *   netPath: string,
 *   maxDepth?: number,
 *   timeBudgetMs?: number
 * }} options
 * @returns {Promise<{ name: string, chooseTurn(state): object|null,
 *                      lastSearchInfo: object|null }>}
 */
export async function createNnueSearchPlayer(rng, { netPath, maxDepth = 64, timeBudgetMs = 1000 } = {}) {
  if (!netPath) throw new Error("createNnueSearchPlayer: netPath is required");
  const jsonPath = normalizeNetPath(netPath);
  const net = await loadNetwork(jsonPath);
  const name = `nnue(${basename(jsonPath)},d${maxDepth},${timeBudgetMs}ms)`;
  return createNnueSearchPlayerFromNet(rng, net, { maxDepth, timeBudgetMs, name });
}
