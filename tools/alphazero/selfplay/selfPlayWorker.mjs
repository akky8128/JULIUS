/**
 * selfPlayWorker.mjs — AlphaZero P4 (PLAN §5.2, §6): MCTS 自己対戦データ生成。
 *
 * 現行 champion net で MCTS 自己対戦し、マイクロ状態ごと1レコードを JSONL 出力する。
 * P3 で修正した mcts.mjs（viableActions / 終端ロジック）をそのまま流用する。
 *
 * 自己対戦モード（評価ゲートとは別。P3 の決定的 OFF は維持）:
 *   - Dirichlet ノイズ（ルートのみ, α=10/平均合法手数, ε=0.25）
 *   - 温度スケジュール（最初 ~16 マイクロ手 τ=1.0 → 以降 0）
 *   - playout cap randomization（任意, 25% フル / 75% 低 sim）
 *
 * ★致命的注意（z の局面ごと手番視点符号, PLAN §6.2）★
 *   z は対局の最終結果を各レコードの「そのマイクロ状態の mover 視点」で付与する:
 *     z = (winner === state.mover) ? +1 : -1  （maxPlies 引き分けは 0.5・統計除外）
 *   π（policyDist）と z は必ず同一対局から取る（NNUE Stage0 の教師矛盾を構造的に防ぐ核心）。
 */

import { generateTurns } from "../../../js/ai/moveGen.js";
import { normalizeBoard } from "../../../js/gameLogic.js";
import {
  NUM_SYMMETRIES,
  extractFeatures,
  transformState,
} from "../../../js/ai/nnue/features.js";
import {
  initTurnState,
  applyMicroAction,
  NUM_MICRO_ACTIONS,
} from "../microActions.mjs";
import {
  TURN_CONTEXT_DIM,
  extractTurnContext,
  transformTurnContext,
  transformActionIndex,
} from "../azFeatures.mjs";
import {
  makeNode,
  expandNode,
  simulate,
  bestActionByVisits,
  viableActions,
  DEFAULT_C_PUCT,
} from "../mcts.mjs";

// loop7 A2: 引き分けの価値を負に。旧 0.5 は train.py で value目標 y=(z+1)/2=0.75 となり、
// 「膠着→引き分け」を勝率75%の好局面として学習させ、決着挑戦(負け-1のリスク)を回避する崩壊
// 誘因の根源だった。-0.5 → y=0.25(不利)。自己対戦互角では決着挑戦のEV≈0(y=0.5)が膠着より
// 優位になり、盤上ループの価値的魅力を除去する。
export const DRAW_Z = -0.5;
export const TEMPERATURE_MICRO_STEPS = 16; // (旧) globalMicroStep 基準。loop6以降は未使用。
// ── loop6 A1: 温度スケジュール（ply=チェス片手番 基準）──
// 序盤 ply < OPENING_PLIES は τ=1.0 で召喚/排除の陣形を多様化（ウケジャ陣形確定 ~6手をカバー）。
// 以降は τ=0 に落とさず SUSTAINED_TEMPERATURE を最後まで維持し、中盤の完全貪欲化＝同一手固執→
// 盤上ループ→引き分け支配、という崩壊機序を断つ。
export const OPENING_PLIES = 6;
export const SUSTAINED_TEMPERATURE = 0.25;
export const DEFAULT_EPSILON = 0.25;
export const DEFAULT_DIRICHLET_SCALE = 10; // α = SCALE / 平均合法手数

// ───────────────────────── z 符号（最重要・専用テストあり）─────────────────────────

/**
 * マイクロ状態の mover 視点で最終結果 z を返す。
 * @param {string|null} winner - "white"|"black"|null(引き分け)
 * @param {string} mover - そのマイクロ状態の手番プレイヤー
 * @returns {number} winner===mover ? +1 : -1、引き分けは DRAW_Z(0.5)
 */
export function computeZ(winner, mover) {
  if (winner === null) return DRAW_Z;
  return winner === mover ? 1 : -1;
}

// ───────────────────────── policyDist（viable 上で正規化・マスク）─────────────────────────

/**
 * ルートの訪問数を viable アクション上で正規化した policyDist(81) を返す。
 * root.legal は expandNode で viable に絞り込まれている前提（非合法/非viable=0）。合計1。
 * 訪問ゼロ（sims=0）の場合は P（事前分布）にフォールバックする。
 */
export function visitPolicyDist(root) {
  const dist = new Float64Array(NUM_MICRO_ACTIONS);
  let total = 0;
  for (const a of root.legal) total += root.N[a];
  if (total > 0) {
    for (const a of root.legal) dist[a] = root.N[a] / total;
  } else if (root.P) {
    // 訪問ゼロ時は事前分布 P（すでに viable 上で softmax 済み）にフォールバック
    for (const a of root.legal) dist[a] = root.P[a];
  }
  return dist;
}

/** policyDist(81) を D4 対称変換 sym で写す（transformActionIndex で 81 要素を置換）。 */
export function transformPolicyDist(dist, sym, boardSize) {
  const out = new Float64Array(NUM_MICRO_ACTIONS);
  for (let a = 0; a < NUM_MICRO_ACTIONS; a++) {
    if (dist[a] !== 0) out[transformActionIndex(a, sym, boardSize)] = dist[a];
  }
  return out;
}

// ───────────────────────── Dirichlet ノイズ（ルートのみ）─────────────────────────

function sampleNormal(rng) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Gamma(alpha,1) 標本（Marsaglia-Tsang, alpha<1 はブースト）。 */
function sampleGamma(alpha, rng) {
  if (alpha < 1) {
    return sampleGamma(alpha + 1, rng) * Math.pow(Math.max(rng(), 1e-12), 1 / alpha);
  }
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      x = sampleNormal(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(Math.max(u, 1e-12)) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** 対称 Dirichlet(alpha) を n 次元でサンプルする。 */
export function sampleDirichlet(n, alpha, rng) {
  const g = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    g[i] = sampleGamma(alpha, rng);
    sum += g[i];
  }
  if (sum === 0) return g.map(() => 1 / n);
  return g.map((x) => x / sum);
}

/**
 * ルートの事前分布 P にのみ Dirichlet ノイズを混ぜる（ルート限定・PLAN §5.2）。
 * P'[a] = (1-ε)P[a] + ε·η[a], η ~ Dir(α), α = scale / |legal|。
 * root は expandNode 済みであること（root.P / root.legal が確定）。
 */
export function applyDirichletNoise(root, { epsilon = DEFAULT_EPSILON, scale = DEFAULT_DIRICHLET_SCALE, rng }) {
  const n = root.legal.length;
  if (n === 0 || !root.P) return;
  const alpha = scale / n;
  const noise = sampleDirichlet(n, alpha, rng);
  for (let i = 0; i < n; i++) {
    const a = root.legal[i];
    root.P[a] = (1 - epsilon) * root.P[a] + epsilon * noise[i];
  }
}

// ───────────────────────── 1マイクロ決定（探索 + 記録）─────────────────────────

/**
 * 1マイクロ状態で MCTS を回し、policyDist と選択アクションを返す。
 * @returns {{ policyDist: Float64Array, action: number, sims: number }}
 */
export function searchMicro(ctx, mover, boardSize, net, opts) {
  const {
    cPuct = DEFAULT_C_PUCT,
    simulations,
    dirichlet = true,
    epsilon = DEFAULT_EPSILON,
    dirichletScale = DEFAULT_DIRICHLET_SCALE,
    temperature = 0,
    rng = Math.random,
  } = opts;

  const root = makeNode(ctx, mover, boardSize);
  // 事前展開（root.P / viable な root.legal を確定）してから Dirichlet をルートに混ぜる。
  expandNode(root, net);
  if (dirichlet) applyDirichletNoise(root, { epsilon, scale: dirichletScale, rng });

  let sims = 0;
  for (; sims < simulations; sims++) simulate(root, net, cPuct);

  const policyDist = visitPolicyDist(root);
  const action = bestActionByVisits(root, { temperature, rng });
  return { policyDist, action, sims };
}

// ───────────────────────── 1局の自己対戦 ─────────────────────────

function nextPlayer(p) {
  return p === "white" ? "black" : "white";
}

function emptyBoard(size) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => []));
}

/**
 * 1局を自己対戦し、マイクロ状態ごとの「base レコード（z 未確定）」列と winner を返す。
 * base レコードは D4 展開前の sym=0 情報のみ持つ。
 */
export function playSelfPlayGame(net, opts) {
  const {
    size = 4,
    maxPlies = 200,
    simulations = 100,
    lowSimulations = null,
    playoutCapProb = 0, // 0 で無効。>0 で確率 playoutCapProb で simulations、残りは lowSimulations
    openingPlies = OPENING_PLIES,       // loop6 A1: この ply 未満は τ=1.0
    sustainedTemperature = SUSTAINED_TEMPERATURE, // loop6 A1: 以降も維持する τ
    rng = Math.random,
  } = opts;

  let board = normalizeBoard(emptyBoard(size), size);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";
  let ply = 0;
  let globalMicroStep = 0;
  const baseRecords = [];
  let winner = null;

  while (ply < maxPlies) {
    const state = {
      board: normalizeBoard(board, size),
      summonCounts: { ...summonCounts },
      currentPlayer,
      boardSize: size,
    };
    if (generateTurns(state).length === 0) {
      winner = nextPlayer(currentPlayer);
      break;
    }

    let ctx = initTurnState(state);
    let microStep = 0;
    for (;;) {
      const legal = viableActions(ctx);
      if (legal.length === 0) {
        throw new Error(`selfPlay: no viable micro-action (phase=${ctx.phase})`);
      }
      // 温度スケジュール(loop6 A1): 序盤 ply < openingPlies は τ=1.0、以降も τ を落とさず維持。
      const temperature = ply < openingPlies ? 1.0 : sustainedTemperature;
      // playout cap randomization（任意）
      let sims = simulations;
      if (playoutCapProb > 0 && lowSimulations != null) {
        sims = rng() < playoutCapProb ? simulations : lowSimulations;
      }

      const microState = {
        board: ctx.board,
        summonCounts: ctx.summonCounts,
        currentPlayer,
        boardSize: size,
      };
      const { indices, scalars } = extractFeatures(microState, currentPlayer);
      const turnContext = Array.from(extractTurnContext(ctx));

      if (legal.length === 1) {
        // 分岐なし: 探索せず policyDist を one-hot にして記録し即適用
        const dist = new Float64Array(NUM_MICRO_ACTIONS);
        dist[legal[0]] = 1;
        baseRecords.push({
          ply, microStep, mover: currentPlayer, indices, scalars, turnContext,
          policyDist: Array.from(dist), _microState: microState,
        });
        ctx = applyMicroAction(ctx, legal[0]);
      } else {
        const { policyDist, action } = searchMicro(ctx, currentPlayer, size, net, {
          simulations: sims,
          temperature,
          rng,
          dirichlet: true,
        });
        baseRecords.push({
          ply, microStep, mover: currentPlayer, indices, scalars, turnContext,
          policyDist: Array.from(policyDist), _microState: microState,
        });
        ctx = applyMicroAction(ctx, action);
      }
      microStep++;
      globalMicroStep++;
      if (ctx.done) {
        board = normalizeBoard(ctx.board, size);
        summonCounts = ctx.summonCounts;
        break;
      }
    }

    currentPlayer = nextPlayer(currentPlayer);
    ply++;
  }

  return { baseRecords, winner, plies: ply, summonCounts: { ...summonCounts } };
}

// ───────────────────────── データセット生成（D4展開 + z付与）─────────────────────────

/**
 * 複数局を自己対戦して最終レコード列（D4×8展開済み・z付与済み）を返す。純関数寄り。
 */
export function buildSelfPlayDataset(net, config) {
  const { games, size = 4, augment = 8, seed = 1, pidOffset = 0 } = config;
  const rng = mulberry32(seed);
  const records = [];
  let nextPid = pidOffset;
  let winW = 0, winB = 0, draws = 0;
  let totalMicro = 0;
  // loop6 D: 崩壊監視指標。決着率・平均ply・平均召喚数を meta に残す。
  let sumPlies = 0, sumSummon = 0;

  for (let g = 0; g < games; g++) {
    const gameRng = mulberry32((rng() * 0xffffffff) >>> 0);
    const { baseRecords, winner, plies, summonCounts } = playSelfPlayGame(net, { ...config, size, rng: gameRng });
    if (winner === "white") winW++; else if (winner === "black") winB++; else draws++;
    sumPlies += plies;
    sumSummon += (summonCounts.white || 0) + (summonCounts.black || 0);

    // pid はターン単位（同一 ply の全 microStep・全 sym で共有）
    const pidByPly = new Map();
    for (const base of baseRecords) {
      totalMicro++;
      if (!pidByPly.has(base.ply)) pidByPly.set(base.ply, nextPid++);
      const pid = pidByPly.get(base.ply);
      const z = computeZ(winner, base.mover); // ★手番視点符号★
      const dist0 = Float64Array.from(base.policyDist);

      const symCount = augment === 8 ? NUM_SYMMETRIES : 1;
      for (let sym = 0; sym < symCount; sym++) {
        let indices, turnContext, policyDist;
        if (sym === 0) {
          indices = base.indices;
          turnContext = base.turnContext;
          policyDist = base.policyDist;
        } else {
          // sym!=0 は盤面から再構成する必要があるため、base に元 microState を保持する。
          const st = transformState(base._microState, sym);
          indices = extractFeatures(st, base.mover).indices;
          turnContext = Array.from(transformTurnContext(Float64Array.from(base.turnContext), sym, size));
          policyDist = Array.from(transformPolicyDist(dist0, sym, size));
        }
        records.push({
          pid, ply: base.ply, microStep: base.microStep, sym,
          size, pov: base.mover,
          indices, scalars: base.scalars, turnContext, policyDist,
          z,
        });
      }
    }
  }

  const meta = {
    format: "az-selfplay-v1",
    turnContextDim: TURN_CONTEXT_DIM,
    numGames: games,
    numRecords: records.length,
    numPositions: nextPid - pidOffset,
    numMicroSteps: totalMicro,
    pidRange: [pidOffset, nextPid],
    resultDistribution: { white: winW, black: winB, draw: draws },
    drawRate: games > 0 ? draws / games : 0,
    avgPlies: games > 0 ? sumPlies / games : 0,
    avgSummon: games > 0 ? sumSummon / games : 0,
    drawPolicy: `maxPlies 到達は z=${DRAW_Z}（loop7 A2: 引き分けペナルティ）`,
  };
  return { records, meta };
}

// ───────────────────────── CLI（明示実行のみ・無人ループではない）─────────────────────────

async function main() {
  const fs = await import("node:fs");
  const path = await import("node:path");
  const { loadNetwork } = await import("../../../js/ai/nnue/network.js");

  const argv = process.argv.slice(2);
  const get = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
  const netPath = get("--net", "models/genAZ001.json");
  const games = parseInt(get("--games", "2"), 10);
  const seed = parseInt(get("--seed", "1"), 10);
  const simulations = parseInt(get("--sims", "60"), 10);
  const maxPlies = parseInt(get("--maxPlies", "200"), 10);
  const augment = parseInt(get("--augment", "8"), 10);
  const pidOffset = parseInt(get("--pidOffset", "0"), 10);
  const openingPlies = parseInt(get("--opening-plies", String(OPENING_PLIES)), 10);
  const sustainedTemperature = parseFloat(get("--sustained-temp", String(SUSTAINED_TEMPERATURE)));
  // 盤面サイズ: --size 明示 > 環境変数 UL_BOARD_SIZE > 4（本番既定）。
  const size = parseInt(get("--size", process.env.UL_BOARD_SIZE || "4"), 10);
  const out = get("--out", "tools/alphazero/data/selfplay/sp000.jsonl");

  // --net random / none で「一様事前確率＋中立value」のネットを使い、seedネット無しの
  // 真のscratch round1 ブートストラップを可能にする（PLAN: 3x3先手有利検証 round1）。
  const net = (netPath === "random" || netPath === "none")
    ? { evaluatePolicy: () => ({ value: 0, policyLogits: new Float64Array(NUM_MICRO_ACTIONS) }) }
    : await loadNetwork(netPath);
  console.log(`self-play: net=${netPath} size=${size} games=${games} sims=${simulations} seed=${seed} openingPlies=${openingPlies} sustainedTemp=${sustainedTemperature}`);
  const t0 = performance.now();
  const { records, meta } = buildSelfPlayDataset(net, { games, size, augment, seed, simulations, maxPlies, pidOffset, openingPlies, sustainedTemperature });
  const elapsed = performance.now() - t0;

  fs.mkdirSync(path.dirname(out), { recursive: true });
  const body = records.map((r) => JSON.stringify(r)).join("\n") + (records.length ? "\n" : "");
  fs.writeFileSync(out, body, "utf8");
  fs.writeFileSync(out.replace(/\.jsonl$/, ".meta.json"), JSON.stringify(meta, null, 2), "utf8");
  console.log(`records=${meta.numRecords} positions=${meta.numPositions} result=${JSON.stringify(meta.resultDistribution)} ${(elapsed / 1000).toFixed(1)}s → ${out}`);
}

const isDirectRun = process.argv[1] && process.argv[1].endsWith("selfPlayWorker.mjs");
if (isDirectRun) {
  main().catch((e) => { console.error(e); process.exitCode = 1; });
}

// mulberry32（genData.mjs/genAZData.mjs と同一）
export function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
