// CPU対戦の「IDA v1」モデル(gen137 NNUE)の着手をサーバー側で計算するCloud Function。
// 重み(functions/models/gen137.*)はクライアントには一切送信せず、選んだ手だけを返す。
import {onCall, HttpsError} from "firebase-functions/v2/https";
import path from "path";
import {fileURLToPath} from "url";
import {maxSummonsFor, normalizeBoard} from "./gameLogic.js";
import {findBestTurn} from "./ai/search.js";
import {loadNetwork} from "./ai/nnue/network.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const NET_PATH = path.join(__dirname, "models", "gen137.json");

// strength ごとの探索予算。クライアントからは列挙値のみ受け取り、
// maxDepth/timeBudgetMsを直接指定させない(コスト濫用防止)。
// sampleMargin: 「最善手の評価値 - sampleMargin」以上のルート候補だけを対象に
// softmaxサンプリングする絶対マージン(評価値の絶対単位、js/ai/search.js 参照)。
// 0だと常に最善手固定で単調になるため、最善手からの損失を sampleMargin 以内に抑えつつ
// 互角に近い候補の中から確率的に選び、対局ごとの多様性を出す。相対スプレッド方式と違い
// 鋭い局面で大損しない。fast(弱め)は多様性を広く、deep(強め)は狭くして棋力を保つ。
const STRENGTH_CONFIG = {
  fast: {maxDepth: 64, timeBudgetMs: 500, sampleMargin: 0.05},
  deep: {maxDepth: 64, timeBudgetMs: 5000, sampleMargin: 0.015},
};

let netPromise = null;
function getNet() {
  if (!netPromise) {
    netPromise = loadNetwork(NET_PATH);
  }
  return netPromise;
}

export const requestIdaMove = onCall(
  {region: "asia-southeast1", memory: "512MiB", timeoutSeconds: 30},
  async (request) => {
    const {board, summonCounts, currentPlayer, boardSize, strength} = request.data || {};

    if (!Number.isInteger(boardSize) || boardSize < 3 || boardSize > 8) {
      throw new HttpsError("invalid-argument", "boardSize must be an integer between 3 and 8.");
    }
    if (currentPlayer !== "white" && currentPlayer !== "black") {
      throw new HttpsError("invalid-argument", "currentPlayer must be \"white\" or \"black\".");
    }
    const strengthConfig = STRENGTH_CONFIG[strength];
    if (!strengthConfig) {
      throw new HttpsError("invalid-argument", "strength must be \"fast\" or \"deep\".");
    }
    if (!Array.isArray(board) || board.length !== boardSize) {
      throw new HttpsError("invalid-argument", "board does not match boardSize.");
    }
    const maxSummons = maxSummonsFor(boardSize);
    const sc = summonCounts || {};
    const scWhite = Number(sc.white);
    const scBlack = Number(sc.black);
    if (
      !Number.isInteger(scWhite) || scWhite < 0 || scWhite > maxSummons ||
      !Number.isInteger(scBlack) || scBlack < 0 || scBlack > maxSummons
    ) {
      throw new HttpsError("invalid-argument", "summonCounts is invalid.");
    }

    const state = {
      board: normalizeBoard(board, boardSize),
      summonCounts: {white: scWhite, black: scBlack},
      currentPlayer,
      boardSize,
    };

    const net = await getNet();

    let result;
    try {
      result = findBestTurn(state, {
        maxDepth: strengthConfig.maxDepth,
        timeBudgetMs: strengthConfig.timeBudgetMs,
        sampleMargin: strengthConfig.sampleMargin,
        rng: Math.random,
        evalFn: (s, p) => net.evaluateState(s, p),
      });
    } catch (err) {
      throw new HttpsError("internal", "Search failed: " + (err.message || String(err)));
    }

    if (!result.turn) {
      return {turn: null};
    }

    return {
      turn: {
        actions: result.turn.actions,
        board: result.turn.board,
        summonCounts: result.turn.summonCounts,
      },
      info: {
        depthReached: result.depthReached,
        nodes: result.nodes,
        elapsedMs: result.elapsedMs,
      },
    };
  },
);
