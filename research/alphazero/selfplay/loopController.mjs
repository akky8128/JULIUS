/**
 * loopController.mjs — 真のAlphaZero型 自己対戦ループ統括（★継続更新・gating無し★）。
 *
 * 1周 = selfplay(最新net) → buffer 追記 → (cadence) サンプル → train 起動 →
 *        新net を **無条件に最新net として採用**（採否判定・champion概念・2標本zは無し）。
 *
 * ★依存注入設計★: selfPlay / train を関数として受け取り、mock で差し替え可能。
 *   これにより「起動できる環境」を実計算ゼロの mock で端から端まで検証できる。
 *
 * ★無人起動しない★: デーモン化・自動ループ開始はしない。runLoop は明示的に呼ばれ、
 *   指定 rounds だけ回して return する。非 dryRun は allowRealRun:true を明示必須。
 *
 * 状態(manifest): { net: <最新単一net>, round, history: [ ... ] }。
 * gating（champion/acceptStreak/decideAcceptance）は廃止した。強さ追跡は別途 benchmark.mjs が
 * 記録専用に行い、採否には一切使わない。
 */

import fs from "node:fs";
import path from "node:path";

function loopManifestPath(dir) {
  return path.join(dir, "loop_manifest.json");
}

/** ループ状態を読み込む（resume）。無ければ初期状態 { net: seedNet, round: 0, history: [] }。 */
export function loadLoopState(dir, seedNet) {
  const mp = loopManifestPath(dir);
  if (fs.existsSync(mp)) {
    const s = JSON.parse(fs.readFileSync(mp, "utf8"));
    // 継続更新schemaへ整合（古い gated manifest を読んでも net を復元）。
    if (!s.net) s.net = s.champion || seedNet;
    if (!Array.isArray(s.history)) s.history = [];
    if (typeof s.round !== "number") s.round = 0;
    return { net: s.net, round: s.round, history: s.history };
  }
  return { net: seedNet, round: 0, history: [] };
}

export function saveLoopState(dir, state) {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(loopManifestPath(dir), JSON.stringify(state, null, 2), "utf8");
}

/**
 * 継続更新ループを rounds 周だけ回す（依存注入）。gating は無く、毎周 net を無条件更新する。
 *
 * @param {object} deps
 *   - selfPlay({ net, round, seed }) => { shardPath, states }
 *   - appendToBuffer({ shardPath, round }) => { totalStates }
 *   - sampleBatch({ round, n }) => { batchPath, sampled }   // cadence 時のみ
 *   - train({ batchPath, round, net }) => { checkpointPath }  // 新net（無条件採用）
 * @param {object} opts
 *   - dir, rounds(有限・必須), seedNet(初期net), trainEveryRounds=1, sampleSize, seedBase,
 *     dryRun=false, allowRealRun=false, log
 * @returns {object} 最終ループ状態
 */
export function runLoop(deps, opts) {
  const {
    dir,
    rounds,
    seedNet,
    initialChampion, // 後方互換エイリアス（旧呼び出し）
    trainEveryRounds = 1,
    sampleSize = 4096,
    seedBase = 1,
    dryRun = false,
    allowRealRun = false,
    log = console.log,
  } = opts;

  const seed = seedNet ?? initialChampion;

  if (!Number.isInteger(rounds) || rounds <= 0) {
    throw new Error("runLoop: rounds は正の整数（有限）である必要があります（無人ループ禁止）");
  }

  // ★事前フライトガード★: 非 dryRun は selfPlay/train を1回でも呼ぶ「前」に allowRealRun 必須。
  if (!dryRun && !allowRealRun) {
    throw new Error(
      "runLoop: 非 dryRun の実行には allowRealRun:true が必須です（うっかり重い自己対戦/train 起動を防止）。" +
      " ループ本体（selfPlay/train）は一切実行していません。"
    );
  }

  const state = loadLoopState(dir, seed);
  const startRound = state.round;

  for (let i = 0; i < rounds; i++) {
    const round = startRound + i + 1;
    state.round = round;

    // 1) 自己対戦（最新 net で）
    const sp = deps.selfPlay({ net: state.net, round, seed: seedBase + round });
    // 2) バッファ追記
    const buf = deps.appendToBuffer({ shardPath: sp.shardPath, round });

    if (dryRun) {
      state.history.push({ round, dryRun: true, states: sp.states, totalStates: buf.totalStates });
      log(`[loop][dry] round=${round} states=${sp.states} bufferTotal=${buf.totalStates}`);
      saveLoopState(dir, state);
      continue;
    }

    // cadence: trainEveryRounds ごとに学習・advance
    if (round % trainEveryRounds !== 0) {
      state.history.push({ round, trained: false, totalStates: buf.totalStates });
      saveLoopState(dir, state);
      continue;
    }

    // 3) サンプル → 4) 学習 → 5) 無条件advance（gating無し）
    const batch = deps.sampleBatch({ round, n: sampleSize });
    const trained = deps.train({ batchPath: batch.batchPath, round, net: state.net });
    const prevNet = state.net;
    state.net = trained.checkpointPath; // ★無条件に最新net として採用★
    state.history.push({
      round, net: state.net, prevNet, totalStates: buf.totalStates,
    });
    log(`[loop] round=${round} advance ${path.basename(prevNet)} → ${path.basename(state.net)} (unconditional)`);
    saveLoopState(dir, state);
  }

  return state;
}

// ───────────────────────── 実依存ファクトリ（配線のみ・自動実行しない）─────────────────────────

/**
 * 実コンポーネント（自己対戦=Node, 学習=train.py 子プロセス）を配線した deps を返す。
 * **この関数は wiring を返すだけで、runLoop に渡して明示実行しない限り実計算は起きない**。
 * 継続更新のため train 後の net は無条件採用（評価ゲート無し）。強さ追跡は benchmark.mjs が別途行う。
 */
export function makeRealDeps(cfg) {
  return {
    async selfPlay({ net, round, seed }) {
      const { loadNetwork } = await import("../../../core/nnue/network.js");
      const sp = await import("./selfPlayWorker.mjs");
      const fs = await import("node:fs");
      const path = await import("node:path");
      const network = await loadNetwork(net);
      const { records, meta } = sp.buildSelfPlayDataset(network, {
        games: cfg.games, size: 4, augment: 8, seed, simulations: cfg.sims, maxPlies: cfg.maxPlies,
        pidOffset: round * 1_000_000,
      });
      fs.mkdirSync(cfg.dataDir, { recursive: true });
      const shardPath = path.join(cfg.dataDir, `sp_round${round}.jsonl`);
      fs.writeFileSync(shardPath, records.map((r) => JSON.stringify(r)).join("\n") + "\n");
      return { shardPath, states: meta.numRecords };
    },
    async appendToBuffer({ shardPath, round }) {
      const rb = await import("./replayBuffer.mjs");
      const buf = rb.openBuffer(cfg.bufferDir, { capacityStates: cfg.capacityStates ?? 500000 });
      buf.manifest.round = round;
      rb.appendShard(buf, shardPath, { round });
      return { totalStates: buf.manifest.totalStates };
    },
    async sampleBatch({ round, n }) {
      const rb = await import("./replayBuffer.mjs");
      const path = await import("node:path");
      const buf = rb.openBuffer(cfg.bufferDir, { capacityStates: cfg.capacityStates ?? 500000 });
      const batchPath = path.join(cfg.dataDir, `batch_round${round}.jsonl`);
      rb.sampleToFile(buf, n, batchPath, Math.random);
      return { batchPath, sampled: n };
    },
    async train({ batchPath, round, net }) {
      const { spawnSync } = await import("node:child_process");
      const path = await import("node:path");
      const tag = `genAZ_r${round}`;
      const initBin = net ? net.replace(/\.json$/, ".bin") : null;
      const args = [cfg.python || "research/nnue/train.py", "--data", batchPath, "--out-dir", cfg.outDir,
        "--gen-tag", tag, "--policy-head", "--stack-encoder-dim", "16",
        "--hidden1", "96", "--hidden2", "24", "--hidden3", "24", "--device", "auto",
        ...(initBin ? ["--init-bin", initBin] : [])];
      const res = spawnSync("python3", args, { stdio: "inherit" });
      if (res.status !== 0) throw new Error(`train.py failed (round ${round})`);
      return { checkpointPath: path.join(cfg.outDir, `${tag}.json`) };
    },
  };
}

// ───────────────────────── CLI（既定 mock/dry・無人ループ禁止）─────────────────────────

function makeMockDeps() {
  let total = 0;
  return {
    selfPlay: ({ round }) => ({ shardPath: `/mock/shard_${round}.jsonl`, states: 100 }),
    appendToBuffer: () => { total += 100; return { totalStates: total }; },
    sampleBatch: ({ round, n }) => ({ batchPath: `/mock/batch_${round}.jsonl`, sampled: n }),
    train: ({ round }) => ({ checkpointPath: `/mock/ckpt_${round}.json` }),
  };
}

async function main() {
  const argv = process.argv.slice(2);
  const get = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
  const has = (k) => argv.includes(k);
  const rounds = parseInt(get("--rounds", "0"), 10);
  const dir = get("--dir", "research/alphazero/data/loop2");
  const seedNet = get("--seed-net", "research/models/genAZ001.json");
  const dryRun = has("--dry-run");
  const mock = has("--mock");

  if (!Number.isInteger(rounds) || rounds <= 0) {
    console.error("使い方: --rounds N(有限・正) [--mock] [--dry-run] [--dir ...] [--seed-net ...]");
    console.error("安全のため無限/無人ループは不可。--mock で実計算ゼロの配線検証、--dry-run で selfPlay 配線のみ。");
    process.exitCode = 1;
    return;
  }
  if (!mock && !dryRun) {
    console.error("実モード（実計算）は本 CLI からは起動しません（無人ループ禁止）。--mock か --dry-run を指定してください。");
    console.error("実運用は runRound.mjs（1周駆動）をオーケストレータから明示的に繰り返してください。");
    process.exitCode = 1;
    return;
  }
  const deps = mock ? makeMockDeps() : {
    selfPlay: ({ round }) => ({ shardPath: `/dry/shard_${round}.jsonl`, states: 0 }),
    appendToBuffer: () => ({ totalStates: 0 }),
    sampleBatch: () => ({ batchPath: "/dry/batch.jsonl", sampled: 0 }),
    train: () => ({ checkpointPath: "/dry/ckpt.json" }),
  };
  const state = runLoop(deps, { dir, rounds, seedNet, dryRun, allowRealRun: mock });
  console.log(`[loop] done. rounds=${state.round} net=${state.net}`);
}

const isDirectRun = process.argv[1] && process.argv[1].endsWith("loopController.mjs");
if (isDirectRun) {
  main().catch((e) => { console.error(e); process.exitCode = 1; });
}
