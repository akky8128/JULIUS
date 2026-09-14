/**
 * sync-core.mjs — core/（ルール・AIの正本）を JuliUs. のデプロイ対象へコピーする。
 *
 * Firebase Hosting は public/ の外を配信できず、Cloud Functions も functions/ しか
 * アップロードしないため、正本 core/ をそれぞれの配下へ複製する。
 * 生成先は .gitignore 済みで、手で編集してはいけない（毎回削除して作り直す）。
 *
 *   core/**  →  julius/public/js/core/**
 *   core/**  →  julius/functions/core/**
 *
 * USAGE:
 *   node julius/scripts/sync-core.mjs          # コピー
 *   node julius/scripts/sync-core.mjs --check  # 生成物が正本と一致するか検査（不一致で exit 1）
 *
 * firebase.json の predeploy から自動実行される。エミュレーター起動前やテスト前にも実行すること。
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const JULIUS_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CORE_DIR = path.resolve(JULIUS_DIR, "..", "core");
const TARGETS = [
  path.join(JULIUS_DIR, "public", "js", "core"),
  path.join(JULIUS_DIR, "functions", "core"),
];
const EXCLUDED_DIRS = new Set(["test"]);
const SOURCE_EXT = /\.m?js$/;

function banner(relPath) {
  return `// AUTO-GENERATED from core/${relPath} by julius/scripts/sync-core.mjs — DO NOT EDIT\n`;
}

/** core/ 配下のコピー対象を { rel, content } の配列で返す */
function collectCoreFiles(dir = CORE_DIR, prefix = "") {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return EXCLUDED_DIRS.has(entry.name) && !prefix ? [] : collectCoreFiles(full, rel);
    }
    if (!SOURCE_EXT.test(entry.name)) return [];
    return [{ rel, content: banner(rel) + fs.readFileSync(full, "utf8") }];
  });
}

function sync(files) {
  for (const target of TARGETS) {
    fs.rmSync(target, { recursive: true, force: true });
    for (const { rel, content } of files) {
      const dest = path.join(target, rel);
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.writeFileSync(dest, content);
    }
    console.log(`sync-core: ${files.length} files -> ${path.relative(process.cwd(), target) || target}`);
  }
}

function check(files) {
  const problems = [];
  for (const target of TARGETS) {
    for (const { rel, content } of files) {
      const dest = path.join(target, rel);
      if (!fs.existsSync(dest)) problems.push(`missing: ${dest}`);
      else if (fs.readFileSync(dest, "utf8") !== content) problems.push(`stale: ${dest}`);
    }
  }
  if (problems.length > 0) {
    console.error(`sync-core --check failed (run: node julius/scripts/sync-core.mjs)\n${problems.join("\n")}`);
    process.exit(1);
  }
  console.log("sync-core --check: up to date");
}

if (!fs.existsSync(CORE_DIR)) {
  console.error(`sync-core: core directory not found: ${CORE_DIR}`);
  process.exit(1);
}
const files = collectCoreFiles();
if (process.argv.includes("--check")) check(files);
else sync(files);
