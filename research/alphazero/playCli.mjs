#!/usr/bin/env node
/**
 * playCli.mjs — AlphaZero移行 P3: エンジン対局CLI＋棋譜入出力。
 *
 * 棋譜記法は既存 core/coords.js（diffToNotation/formatCoord/parseCoord）を
 * そのまま利用し、座標系（a1={r:3,c:0}）と完全一致させる。
 *
 * エンジン指定:
 *   az:<model.json>:<msPerTurn>        AZ(v0)+MCTS
 *   search:<model.json>:<msPerTurn>    NNUE eval + αβ（findBestTurn）
 *   search:<msPerTurn>                 heuristic + αβ
 *   human                              棋譜文字列を標準入力から
 *
 * 使い方:
 *   node research/alphazero/playCli.mjs --white az:research/models/genAZ001.json:120 \
 *        --black search:research/models/gen137.json:120 [--maxPlies 200] [--seed 1] [--quiet]
 */

import { generateTurns } from "../../core/ai/moveGen.js";
import { hashPosition } from "../../core/ai/moveGen.js";
import { findBestTurn } from "../../core/ai/search.js";
import { loadNetwork } from "../../core/nnue/network.js";
import { normalizeBoard, denormalizeBoard, maxSummonsFor } from "../../core/gameLogic.js";
import { diffToNotation, formatCoord } from "../../core/coords.js";
import { createMctsPlayer } from "./mctsPlayer.mjs";

// ───────────────────────── 盤面ユーティリティ ─────────────────────────

export function emptyBoard(size) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => []));
}

function nextPlayer(p) {
  return p === "white" ? "black" : "white";
}

/** 適用前後の（正規化済み）盤面から棋譜トークンを生成する。 */
export function turnToNotation(boardBefore, boardAfter, boardSize) {
  return diffToNotation(
    denormalizeBoard(boardBefore),
    denormalizeBoard(boardAfter),
    boardSize
  );
}

/** ASCII 盤面（上が rank 高、a列が左）。各セルは 最上駒色(大文字=white)+高さ。 */
export function renderBoard(board, boardSize) {
  const lines = [];
  for (let r = 0; r < boardSize; r++) {
    const rank = boardSize - r;
    const cells = [];
    for (let c = 0; c < boardSize; c++) {
      const stack = board[r][c];
      if (!Array.isArray(stack) || stack.length === 0) {
        cells.push(" . ");
      } else {
        const top = stack[stack.length - 1];
        const ch = top === "white" ? "W" : "b";
        cells.push(` ${ch}${stack.length}`);
      }
    }
    lines.push(`${rank} ${cells.join("")}`);
  }
  const files = Array.from({ length: boardSize }, (_, c) => `  ${String.fromCharCode(97 + c)}`).join("");
  lines.push(`  ${files}`);
  return lines.join("\n");
}

// ───────────────────────── 棋譜入力のパース ─────────────────────────

/** 棋譜トークンを (係数, 座標) の原子列に分解する（例 "3a22a3b3" → [{mag:3,coord:"a2"},...]）。 */
export function parseNotationAtoms(token) {
  const atoms = [];
  // 座標の段(rank)は1桁（boardSize<=9 前提。4x4では1..4）。係数だけが複数桁になり得る。
  // rank を \d+ にすると "2c12d1" が c+rank12 と誤読されるため rank は \d（1桁）に固定する。
  const re = /(\d*)([a-z])(\d)/g;
  let m;
  while ((m = re.exec(token)) !== null) {
    atoms.push({ mag: m[1] ? parseInt(m[1], 10) : 1, coord: `${m[2]}${m[3]}` });
  }
  return atoms;
}

/** denormalized 盤面の変化を「減少セル」「増加セル」に分けて (r,c)順で返す。 */
function cellDeltas(beforeD, afterD, boardSize) {
  const len = (b, r, c) => (Array.isArray(b?.[r]?.[c]) ? b[r][c].length : 0);
  const dec = [];
  const inc = [];
  for (let r = 0; r < boardSize; r++) {
    for (let c = 0; c < boardSize; c++) {
      const d = len(afterD, r, c) - len(beforeD, r, c);
      if (d < 0) dec.push({ coord: formatCoord(r, c, boardSize), mag: -d });
      else if (d > 0) inc.push({ coord: formatCoord(r, c, boardSize), mag: d });
    }
  }
  return { dec, inc };
}

/** (係数,座標)原子集合の順不同キー。 */
function atomMultisetKey(atoms) {
  return atoms.map((a) => `${a.mag}${a.coord}`).sort().join(",");
}

/**
 * 棋譜トークン（例 "a1", "c3d3", "3a22a3b3", ファイル語順の "2d1d2c2" 等）を現局面の
 * 合法ターンに突き合わせ、対応する候補を返す。
 *
 * ★グループ内語順非依存★: 記法規約「減った座標全て → 増えた座標全て」の**グループ境界**は
 * 尊重しつつ、各グループ内の語順は問わない（coords.js は (r,c)走査順で emit するが、
 * 添付棋譜ファイルは別の語順を用いるため。js/ 変更不可のためパーサ側で吸収する）。
 * 各候補の実 board 差分（減少/増加セルの多重集合）と、入力トークンを候補の減少セル数で
 * 分割した前半/後半の多重集合が一致するかで判定する。
 * 異なる結果盤面が複数一致する場合のみ曖昧エラー。
 */
export function parseTurnNotation(token, state) {
  const boardSize = state.boardSize;
  const candidates = generateTurns(state);
  const beforeD = denormalizeBoard(state.board);
  const atoms = parseNotationAtoms(token);

  const matches = [];
  const seenHashes = new Set();
  for (const cand of candidates) {
    const { dec, inc } = cellDeltas(beforeD, denormalizeBoard(cand.board), boardSize);
    if (atoms.length !== dec.length + inc.length) continue;
    // 記法規約: 前半=減少セル、後半=増加セル。境界は候補の減少セル数で決まる。
    const tokDec = atoms.slice(0, dec.length);
    const tokInc = atoms.slice(dec.length);
    if (atomMultisetKey(tokDec) === atomMultisetKey(dec) &&
        atomMultisetKey(tokInc) === atomMultisetKey(inc)) {
      const h = hashPosition(cand.board, cand.summonCounts);
      if (!seenHashes.has(h)) {
        seenHashes.add(h);
        matches.push(cand);
      }
    }
  }
  if (matches.length === 0) {
    throw new Error(`棋譜トークン "${token}" に対応する合法手がありません`);
  }
  if (matches.length > 1) {
    throw new Error(`棋譜トークン "${token}" が曖昧です（${matches.length} 通りの異なる結果盤面）`);
  }
  return matches[0];
}

// ───────────────────────── エンジン生成 ─────────────────────────

/**
 * エンジン指定文字列からエージェント {name, chooseTurn(state)} を作る。
 * chooseTurn は {board, summonCounts} を返す。
 */
export async function createEngine(spec, { rng = Math.random } = {}) {
  const parts = spec.split(":");
  const kind = parts[0];

  if (kind === "az") {
    const modelPath = parts[1];
    const ms = parseInt(parts[2] || "120", 10);
    const net = await loadNetwork(modelPath);
    const player = createMctsPlayer(net, { timeBudgetMs: ms, rng });
    return {
      name: `az:${modelPath}:${ms}`,
      chooseTurn(state) {
        const r = player.chooseTurn(state);
        return { board: r.board, summonCounts: r.summonCounts };
      },
    };
  }

  if (kind === "search") {
    // search:<model.json>:<ms>  または  search:<ms>
    let modelPath = null;
    let ms;
    if (parts[1] && (parts[1].includes("/") || parts[1].endsWith(".json"))) {
      modelPath = parts[1];
      ms = parseInt(parts[2] || "120", 10);
    } else {
      ms = parseInt(parts[1] || "120", 10);
    }
    let evalFn = null;
    if (modelPath) {
      const net = await loadNetwork(modelPath);
      evalFn = (s, p) => net.evaluateState(s, p);
    }
    return {
      name: modelPath ? `search:${modelPath}:${ms}` : `search:heuristic:${ms}`,
      chooseTurn(state) {
        const opts = { maxDepth: 3, timeBudgetMs: ms, rng };
        if (evalFn) opts.evalFn = evalFn;
        const res = findBestTurn(state, opts);
        if (!res.turn) throw new Error("search: no legal turn");
        return { board: res.turn.board, summonCounts: res.turn.summonCounts };
      },
    };
  }

  throw new Error(`未知のエンジン指定: ${spec}`);
}

// ───────────────────────── 対局ループ ─────────────────────────

/**
 * 2エンジンで1局を対戦し、勝者・棋譜・最終局面を返す。
 * @returns {{ winner:string|null, moves:string[], plies:number, notation:string }}
 */
export function playGame(whiteEngine, blackEngine, { size = 4, maxPlies = 200 } = {}) {
  const engines = { white: whiteEngine, black: blackEngine };
  let board = normalizeBoard(emptyBoard(size), size);
  let summonCounts = { white: 0, black: 0 };
  let currentPlayer = "white";
  let ply = 0;
  const moves = []; // 各 ply の棋譜トークン
  let winner = null;

  while (ply < maxPlies) {
    const state = {
      board: normalizeBoard(board, size),
      summonCounts: { ...summonCounts },
      currentPlayer,
      boardSize: size,
    };
    const candidates = generateTurns(state);
    if (candidates.length === 0) {
      winner = nextPlayer(currentPlayer);
      break;
    }
    const chosen = engines[currentPlayer].chooseTurn(state);
    const token = turnToNotation(state.board, chosen.board, size);
    moves.push(token);
    board = normalizeBoard(chosen.board, size);
    summonCounts = chosen.summonCounts;
    currentPlayer = nextPlayer(currentPlayer);
    ply++;
  }

  return { winner, moves, plies: ply, notation: formatGameNotation(moves, winner) };
}

/** moves（ply毎トークン）を "1. w b\n2. ..." 形式に整形し、勝ちに #/1-0/0-1 を付す。 */
export function formatGameNotation(moves, winner) {
  const lines = [];
  for (let i = 0; i < moves.length; i += 2) {
    const moveNo = i / 2 + 1;
    const w = moves[i];
    const b = moves[i + 1];
    const isLastW = i === moves.length - 1;
    const isLastB = i + 1 === moves.length - 1;
    let wTok = w;
    let bTok = b || "";
    if (winner && isLastW) wTok += "#";
    if (winner && isLastB) bTok += "#";
    let line = `${moveNo}. ${wTok}${bTok ? " " + bTok : ""}`;
    lines.push(line.trimEnd());
  }
  if (winner) lines.push(winner === "white" ? "1-0" : "0-1");
  return lines.join("\n");
}

// ───────────────────────── main ─────────────────────────

function parseArgs(argv) {
  const args = { white: null, black: null, size: 4, maxPlies: 200, seed: 1, quiet: false };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--white") args.white = argv[++i];
    else if (argv[i] === "--black") args.black = argv[++i];
    else if (argv[i] === "--size") args.size = parseInt(argv[++i], 10);
    else if (argv[i] === "--maxPlies") args.maxPlies = parseInt(argv[++i], 10);
    else if (argv[i] === "--seed") args.seed = parseInt(argv[++i], 10);
    else if (argv[i] === "--quiet") args.quiet = true;
  }
  return args;
}

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.white || !args.black) {
    console.error("使い方: --white <engine> --black <engine>");
    console.error('例: --white az:research/models/genAZ001.json:120 --black search:research/models/gen137.json:120');
    process.exitCode = 1;
    return;
  }
  const rng = mulberry32(args.seed);
  const white = await createEngine(args.white, { rng });
  const black = await createEngine(args.black, { rng });

  console.log(`White: ${white.name}`);
  console.log(`Black: ${black.name}`);
  console.log("─".repeat(50));

  const t0 = performance.now();
  const result = playGame(white, black, { size: args.size, maxPlies: args.maxPlies });
  const elapsed = performance.now() - t0;

  console.log(result.notation);
  console.log("─".repeat(50));
  console.log(`winner: ${result.winner ?? "none(maxPlies)"}  plies: ${result.plies}  ${(elapsed / 1000).toFixed(1)}s`);
}

const isDirectRun = process.argv[1] && process.argv[1].endsWith("playCli.mjs");
if (isDirectRun) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
