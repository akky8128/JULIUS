/**
 * replayBuffer.mjs — AlphaZero P4 (PLAN §6.1): JSONL シャードのリングバッファ。
 *
 * 自己対戦ワーカーが出力する JSONL シャードをディレクトリに蓄積し、直近 N 状態
 * （capacityStates, 初期 50万目安）だけを保持する。容量超過時は古いシャードから退避
 * （削除 or archive）。学習用バッチは保持シャードから一様サンプリングして書き出す。
 * resume 用に manifest.json を永続化する。純関数寄りで、fs 操作は最小限に集約する。
 */

import fs from "node:fs";
import path from "node:path";

const MANIFEST_NAME = "manifest.json";

function manifestPath(dir) {
  return path.join(dir, MANIFEST_NAME);
}

/** JSONL の行数（= 状態数）を数える（末尾改行の有無に頑健）。 */
export function countJsonlLines(filePath) {
  const text = fs.readFileSync(filePath, "utf8");
  if (text.length === 0) return 0;
  let n = 0;
  for (let i = 0; i < text.length; i++) if (text[i] === "\n") n++;
  // 末尾に改行がなければ最後の行を+1
  if (text[text.length - 1] !== "\n") n++;
  return n;
}

/**
 * バッファを開く（無ければ作成）。manifest があれば読み込み resume する。
 * @param {string} dir
 * @param {{capacityStates?: number, archiveDir?: string|null}} opts
 */
export function openBuffer(dir, opts = {}) {
  const { capacityStates = 500000, archiveDir = null } = opts;
  fs.mkdirSync(dir, { recursive: true });
  if (archiveDir) fs.mkdirSync(archiveDir, { recursive: true });
  let manifest;
  const mp = manifestPath(dir);
  if (fs.existsSync(mp)) {
    manifest = JSON.parse(fs.readFileSync(mp, "utf8"));
  } else {
    manifest = { round: 0, capacityStates, totalStates: 0, shards: [] };
  }
  // capacity は開くたびに opts を優先（運用中の変更を許容）
  manifest.capacityStates = capacityStates;
  return { dir, archiveDir, manifest };
}

export function saveManifest(buffer) {
  fs.writeFileSync(manifestPath(buffer.dir), JSON.stringify(buffer.manifest, null, 2), "utf8");
}

/** 容量超過分を古いシャードから退避する（archiveDir があれば move、無ければ削除）。 */
function evictIfNeeded(buffer) {
  const m = buffer.manifest;
  while (m.totalStates > m.capacityStates && m.shards.length > 1) {
    const oldest = m.shards.shift();
    m.totalStates -= oldest.states;
    const src = path.join(buffer.dir, oldest.file);
    if (fs.existsSync(src)) {
      if (buffer.archiveDir) {
        fs.renameSync(src, path.join(buffer.archiveDir, oldest.file));
      } else {
        fs.unlinkSync(src);
      }
    }
  }
}

/**
 * JSONL シャードをバッファに追加する。srcPath の内容をバッファ内へコピーし manifest 更新、
 * 容量超過分を退避する。
 * @returns {{ file: string, states: number }} 追加されたシャード情報
 */
export function appendShard(buffer, srcPath, { round = null } = {}) {
  const states = countJsonlLines(srcPath);
  const base = `shard_${String(buffer.manifest.shards.length).padStart(5, "0")}_${path.basename(srcPath)}`;
  const dest = path.join(buffer.dir, base);
  fs.copyFileSync(srcPath, dest);
  const entry = { file: base, states, round: round ?? buffer.manifest.round };
  buffer.manifest.shards.push(entry);
  buffer.manifest.totalStates += states;
  evictIfNeeded(buffer);
  saveManifest(buffer);
  return entry;
}

/**
 * 保持シャード全体から n 状態を一様サンプリングして outPath に JSONL 書き出しする。
 * reservoir sampling でメモリ O(n)。
 * @param {object} buffer
 * @param {number} n
 * @param {string} outPath
 * @param {() => number} rng - [0,1) 乱数（再現性用）
 * @returns {{ sampled: number, seen: number }}
 */
export function sampleToFile(buffer, n, outPath, rng = Math.random) {
  const reservoir = new Array(n);
  let seen = 0;
  for (const shard of buffer.manifest.shards) {
    const fp = path.join(buffer.dir, shard.file);
    if (!fs.existsSync(fp)) continue;
    const text = fs.readFileSync(fp, "utf8");
    const lines = text.split("\n");
    for (const line of lines) {
      if (line.trim() === "") continue;
      seen++;
      if (seen <= n) {
        reservoir[seen - 1] = line;
      } else {
        const j = Math.floor(rng() * seen);
        if (j < n) reservoir[j] = line;
      }
    }
  }
  const sampled = Math.min(n, seen);
  const body = reservoir.slice(0, sampled).join("\n") + (sampled > 0 ? "\n" : "");
  fs.writeFileSync(outPath, body, "utf8");
  return { sampled, seen };
}

/**
 * loop6 C1: 決着局オーバーサンプリング付きサンプラ。
 * 引き分けレコード（z===DRAW_Z=0.5）が自己対戦の ~87% を占め、決着(z=±1)の教師が薄まる問題に
 * 対処するため、決着:引き分け を decisiveFrac:(1-decisiveFrac) に補正して n 件を構成する。
 * 決着レコードが目標数に満たない場合は残りを引き分けで埋める（不足時フォールバック）。
 * 2ストラタムの reservoir sampling でメモリは O(n)（各 reservoir を n で上限）。
 * @returns {{ sampled, seen, decisive, draw, decisiveAvail, drawAvail }}
 */
export function sampleStratifiedToFile(buffer, n, outPath, rng = Math.random, opts = {}) {
  const { decisiveFrac = 0.5, drawZ = 0.5 } = opts;
  const decRes = new Array(n);
  const drawRes = new Array(n);
  let decSeen = 0, drawSeen = 0;

  const offer = (line, reservoir, seen) => {
    if (seen <= n) {
      reservoir[seen - 1] = line;
    } else {
      const j = Math.floor(rng() * seen);
      if (j < n) reservoir[j] = line;
    }
  };

  for (const shard of buffer.manifest.shards) {
    const fp = path.join(buffer.dir, shard.file);
    if (!fs.existsSync(fp)) continue;
    const text = fs.readFileSync(fp, "utf8");
    for (const line of text.split("\n")) {
      if (line.trim() === "") continue;
      let z;
      try { z = JSON.parse(line).z; } catch { continue; }
      const isDraw = z === drawZ;
      if (isDraw) { drawSeen++; offer(line, drawRes, drawSeen); }
      else { decSeen++; offer(line, decRes, decSeen); }
    }
  }

  const decisiveAvail = Math.min(n, decSeen);
  const drawAvail = Math.min(n, drawSeen);
  let wantDec = Math.min(Math.round(n * decisiveFrac), decisiveAvail);
  let wantDraw = Math.min(n - wantDec, drawAvail);
  // 決着が目標に満たない → 引き分けで埋める（既に上で n-wantDec 上限）。
  // 逆に引き分けが足りない場合は決着を追加で使う。
  if (wantDec + wantDraw < n) {
    wantDec = Math.min(decisiveAvail, n - wantDraw);
  }

  const picked = [];
  for (let i = 0; i < wantDec; i++) picked.push(decRes[i]);
  for (let i = 0; i < wantDraw; i++) picked.push(drawRes[i]);
  // シャッフル（決着ブロックと引き分けブロックが連続しないように）
  for (let i = picked.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [picked[i], picked[j]] = [picked[j], picked[i]];
  }
  const body = picked.join("\n") + (picked.length > 0 ? "\n" : "");
  fs.writeFileSync(outPath, body, "utf8");
  return {
    sampled: picked.length, seen: decSeen + drawSeen,
    decisive: wantDec, draw: wantDraw, decisiveAvail: decSeen, drawAvail: drawSeen,
  };
}

/** 現在の保持状態数・シャード数を返す（テスト・監視用）。 */
export function bufferStats(buffer) {
  return {
    round: buffer.manifest.round,
    totalStates: buffer.manifest.totalStates,
    numShards: buffer.manifest.shards.length,
    capacityStates: buffer.manifest.capacityStates,
  };
}
