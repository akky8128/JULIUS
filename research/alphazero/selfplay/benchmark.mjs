#!/usr/bin/env node
/**
 * benchmark.mjs — 強さ追跡の記録専用ベンチマーク（★採否には一切使わない★）。
 *
 * 指定ネット（既定=manifestの現行net）を代表モデル群（パネル）と対戦させ、
 * <dir>/benchmarks.jsonl に1行 append する:
 *   {round, net, timestamp, results: { "<opponent>": {games,wins,losses,draws,winrate,ciLo,ciHi,msTurnFocal,msTurnOpp} }}
 *
 * winrate と Wilson95%CI は「決着局のみ（wins/(wins+losses)）」で算出（evalGate.mjs 準拠）。
 * 各対戦は --bench-games を procs で分割し色交換つき並列実行（evalWorker の各モードを使用）。
 *
 * パネル（既定）:
 *   heuristic-d16:80   : evalWorker vsHeuristic --hdepth 16 --hms 80
 *   heuristic-d20:120  : evalWorker vsHeuristic --hdepth 20 --hms 120
 *   gen137+ab          : evalWorker vsSearchNet --oppNet research/models/gen137.json --sdepth 32 --sms 120
 *   genAZ001+mcts      : evalWorker headToHead  --netB research/models/genAZ001.json
 *
 * 使い方（単体CLI）:
 *   node research/alphazero/selfplay/benchmark.mjs \
 *     --net research/models/genAZ_r55.json --round 55 --dir research/alphazero/data/loop2 --bench-games 40 --procs 10
 * （--net 省略時は <dir>/loop_manifest.json の net を使う）
 */

import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..", "..");

function arg(k, d) { const i = process.argv.indexOf(k); return i >= 0 ? process.argv[i + 1] : d; }

function run(cmd, args, label) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd: ROOT, stdio: ["ignore", "pipe", "inherit"] });
    let out = "";
    child.stdout.on("data", (d) => { out += d.toString(); });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) reject(new Error(`${label || cmd} exited ${code}`));
      else resolve(out);
    });
  });
}

async function pMapLimit(tasks, concurrency) {
  const results = new Array(tasks.length);
  let idx = 0;
  async function worker() {
    while (idx < tasks.length) { const c = idx++; results[c] = await tasks[c](); }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, tasks.length) }, worker));
  return results;
}

function splitGames(total, procs) {
  const base = Math.floor(total / procs);
  const rem = total % procs;
  return Array.from({ length: procs }, (_, i) => base + (i < rem ? 1 : 0)).filter((g) => g > 0);
}

/** Wilson 95%CI（evalGate.mjs 準拠）。[p, lo, hi] を返す。 */
function wilson(wins, n, z = 1.96) {
  if (n === 0) return [0, 0, 0];
  const p = wins / n;
  const denom = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [p, center - half, center + half];
}

/**
 * パネル定義: 各 opponent の evalWorker 引数（--netA/--games/--seed はシャードごとに付与）。
 * full=false（既定・routine）は高速で安定シグナルの random + heuristic 2種のみ。
 * full=true は gen137+ab / genAZ001+mcts（探索が重く時間がかかる・gen137は結果が振動）も含む。
 * routine で軽量化し、gen137 の確定値は必要時に --full で単発計測する（時間予算の保護）。
 */
function panel(full = false) {
  const routine = [
    { name: "random", args: ["--mode", "vsRandom"], seedBase: 50000 },
    { name: "heuristic-d16:80", args: ["--mode", "vsHeuristic", "--hdepth", "16", "--hms", "80"], seedBase: 100000 },
    { name: "heuristic-d20:120", args: ["--mode", "vsHeuristic", "--hdepth", "20", "--hms", "120"], seedBase: 200000 },
  ];
  if (!full) return routine;
  return [
    ...routine,
    { name: "gen137+ab", args: ["--mode", "vsSearchNet", "--oppNet", "research/models/gen137.json", "--sdepth", "32", "--sms", "120"], seedBase: 300000 },
    { name: "genAZ001+mcts", args: ["--mode", "headToHead", "--netB", "research/models/genAZ001.json"], seedBase: 400000 },
  ];
}

async function benchOne(opp, net, benchGames, procs, sims, round) {
  const shardGames = splitGames(benchGames, procs);
  const tasks = shardGames.map((g, i) => () => run("node", [
    "research/alphazero/selfplay/evalWorker.mjs",
    "--netA", net, "--games", String(g),
    "--seed", String(opp.seedBase + round * 1000 + i), "--sims", String(sims),
    ...opp.args,
  ], `bench ${opp.name} s${i}`));
  const outs = await pMapLimit(tasks, procs);
  const parsed = outs.map((o) => JSON.parse(o.trim()));
  const agg = parsed.reduce((a, r) => ({
    wins: a.wins + r.wins, losses: a.losses + r.losses, draws: a.draws + r.draws,
    msF: a.msF + r.msTurnFocal, msO: a.msO + r.msTurnOpp, n: a.n + 1,
  }), { wins: 0, losses: 0, draws: 0, msF: 0, msO: 0, n: 0 });
  const games = agg.wins + agg.losses + agg.draws;
  const decisive = agg.wins + agg.losses;
  const [p, lo, hi] = wilson(agg.wins, decisive);
  return {
    games, wins: agg.wins, losses: agg.losses, draws: agg.draws,
    winrate: Number(p.toFixed(4)), ciLo: Number(lo.toFixed(4)), ciHi: Number(hi.toFixed(4)),
    msTurnFocal: Number((agg.msF / Math.max(1, agg.n)).toFixed(2)),
    msTurnOpp: Number((agg.msO / Math.max(1, agg.n)).toFixed(2)),
  };
}

async function main() {
  const dir = arg("--dir", "research/alphazero/data/loop2");
  let net = arg("--net", null);
  let round = arg("--round", null);
  const benchGames = parseInt(arg("--bench-games", "40"), 10);
  const procs = parseInt(arg("--procs", "10"), 10);
  const sims = parseInt(arg("--sims", "100"), 10);
  const full = process.argv.includes("--full"); // 既定は routine(random+heuristic2種)。--full で全対戦。

  if (!net || round === null) {
    const mp = path.join(dir, "loop_manifest.json");
    if (fs.existsSync(mp)) {
      const st = JSON.parse(fs.readFileSync(mp, "utf8"));
      if (!net) net = st.net || st.champion;
      if (round === null) round = String(st.round ?? 0);
    }
  }
  if (!net) { console.error("benchmark: --net 未指定かつ manifest から解決できません"); process.exitCode = 1; return; }
  round = parseInt(round ?? "0", 10);

  const log = (m) => console.log(`[bench r${round}] ${m}`);
  log(`net=${net} benchGames=${benchGames} procs=${procs} sims=${sims} full=${full}`);

  const results = {};
  for (const opp of panel(full)) {
    const t = Date.now();
    results[opp.name] = await benchOne(opp, net, benchGames, procs, sims, round);
    const r = results[opp.name];
    log(`${opp.name}: ${r.wins}/${r.wins + r.losses} = ${(r.winrate * 100).toFixed(1)}% ` +
      `CI[${(r.ciLo * 100).toFixed(1)},${(r.ciHi * 100).toFixed(1)}] draws=${r.draws} ` +
      `ms/turn focal=${r.msTurnFocal} opp=${r.msTurnOpp} (${((Date.now() - t) / 1000).toFixed(1)}s)`);
  }

  const line = { round, net, timestamp: new Date().toISOString(), results };
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(path.join(dir, "benchmarks.jsonl"), JSON.stringify(line) + "\n");
  log(`appended → ${path.join(dir, "benchmarks.jsonl")}`);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
