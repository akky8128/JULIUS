/**
 * migrateCurrentNode.mjs
 *
 * One-time migration script: backfills the `current` node for games in Firebase RTDB.
 *
 * USAGE:
 *   node tools/migrateCurrentNode.mjs [--dry-run] [--all]
 *
 * OPTIONS:
 *   --dry-run   Print what would be written without making any DB changes.
 *   --all       Also migrate completed games that are missing `current`.
 *               Default: only migrates games with meta.status === "in_progress" or "waiting".
 *
 * ENVIRONMENT:
 *   GOOGLE_APPLICATION_CREDENTIALS  (required) Path to your Firebase service-account JSON.
 *                                   e.g. export GOOGLE_APPLICATION_CREDENTIALS=~/julius-sa.json
 *   DATABASE_URL                     (optional) Override the RTDB URL.
 *                                   Defaults to: https://julius-online-a5984-default-rtdb.firebaseio.com
 *
 * firebase-admin is loaded via createRequire pointing at functions/package.json so Node
 * resolves it from functions/node_modules/ without any path-hacking.
 */

import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import path from 'path';

// Resolve firebase-admin from functions/node_modules
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const functionsPackageJson = path.resolve(__dirname, '../functions/package.json');
const require = createRequire(functionsPackageJson);
const admin = require('firebase-admin');

// ─── Config ─────────────────────────────────────────────────────────────────

const DEFAULT_DATABASE_URL = 'https://julius-online-a5984-default-rtdb.firebaseio.com';
const DATABASE_URL = process.env.DATABASE_URL || DEFAULT_DATABASE_URL;
const BATCH_SIZE = 50; // number of games per multi-path update call

// ─── Argument parsing ────────────────────────────────────────────────────────

const args = new Set(process.argv.slice(2));
const DRY_RUN = args.has('--dry-run');
const ALL = args.has('--all');

if (DRY_RUN) {
  console.log('[dry-run] No writes will be performed.');
}

// ─── Firebase init ───────────────────────────────────────────────────────────

if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
  console.error(
    'ERROR: GOOGLE_APPLICATION_CREDENTIALS is not set.\n' +
    'Export the path to your service-account JSON, e.g.:\n' +
    '  export GOOGLE_APPLICATION_CREDENTIALS=~/julius-sa.json'
  );
  process.exit(1);
}

admin.initializeApp({ databaseURL: DATABASE_URL });
const db = admin.database();

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Given a `moves` value (JS array or plain object keyed by turn number),
 * return the last entry. Returns null if there are no moves.
 */
function getLastMove(moves) {
  if (!moves || typeof moves !== 'object') return null;

  if (Array.isArray(moves)) {
    // Sparse arrays are possible; find the highest real index
    let last = null;
    for (let i = 0; i < moves.length; i++) {
      if (moves[i] !== undefined && moves[i] !== null) last = moves[i];
    }
    return last;
  }

  // Object keyed by turnNumber (string keys like "0", "1", ...)
  const keys = Object.keys(moves);
  if (keys.length === 0) return null;

  // Sort numerically
  keys.sort((a, b) => Number(a) - Number(b));
  return moves[keys[keys.length - 1]];
}

/**
 * Build the `current` snapshot from the last move and meta.status.
 * Returns null if the last move is missing required fields.
 */
function buildCurrent(lastMove, metaStatus) {
  if (!lastMove) return null;

  const { board, summonCounts, timers, timestamp, currentPlayer, turnNumber } = lastMove;

  if (board === undefined) {
    // Some edge-case entry lacks board — skip
    return null;
  }

  return {
    turnNumber: turnNumber ?? null,
    currentPlayer: currentPlayer ?? null,
    board,
    summonCounts: summonCounts ?? null,
    timers: timers ?? null,
    timestamp: timestamp ?? null,
    // Use meta.status verbatim.
    // Note: the new joinGame Cloud Function will overwrite current/status = "in_progress"
    // when a player joins a "waiting" game, so it's safe to store "waiting" here as-is.
    status: metaStatus,
  };
}

/**
 * Flush an accumulated update object to RTDB.
 */
async function flushBatch(updatePayload) {
  if (Object.keys(updatePayload).length === 0) return;
  if (DRY_RUN) return;
  await db.ref('/').update(updatePayload);
}

// ─── SIGINT handler ───────────────────────────────────────────────────────────

let shuttingDown = false;
process.on('SIGINT', () => {
  console.log('\nSIGINT received — finishing current batch then exiting.');
  shuttingDown = true;
});

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  // Read all games in one shot.
  // At current scale (~1.8 MB, ~260 games) a single once('value') is acceptable
  // and avoids thundering-herd issues with per-game reads.
  console.log(`Reading /games from ${DATABASE_URL} …`);
  const snapshot = await db.ref('/games').once('value');
  const allGames = snapshot.val();

  if (!allGames || typeof allGames !== 'object') {
    console.log('No games found under /games. Nothing to do.');
    process.exit(0);
  }

  const gameIds = Object.keys(allGames);
  console.log(`Scanned ${gameIds.length} games.`);

  let countMigrated = 0;
  let countSkipped = 0;
  let countErrors = 0;

  const activeStatuses = new Set(['in_progress', 'waiting']);

  let batchPayload = {};
  let batchCount = 0;

  for (const id of gameIds) {
    if (shuttingDown) break;

    const game = allGames[id];

    // --- Determine eligibility ---
    const meta = game.meta ?? {};
    const status = meta.status ?? null;
    const hasCurrent = game.current !== undefined && game.current !== null;

    if (hasCurrent) {
      countSkipped++;
      continue;
    }

    const isActive = activeStatuses.has(status);
    const isCompleted = !isActive; // anything else: "completed", "abandoned", etc.

    if (!isActive && !ALL) {
      // Not active and --all not requested — skip
      countSkipped++;
      continue;
    }

    if (isCompleted && !ALL) {
      countSkipped++;
      continue;
    }

    // --- Build current node ---
    let lastMove = null;
    try {
      lastMove = getLastMove(game.moves);
    } catch (err) {
      console.error(`  [${id}] Error reading moves: ${err.message}`);
      countErrors++;
      continue;
    }

    if (!lastMove) {
      // Game has no moves at all (e.g. brand-new waiting game).
      // We cannot reconstruct current without a move. Skip gracefully.
      console.warn(`  [${id}] status=${status} — no moves found, skipping.`);
      countSkipped++;
      continue;
    }

    const current = buildCurrent(lastMove, status);
    if (!current) {
      console.warn(`  [${id}] status=${status} — last move has no board, skipping.`);
      countSkipped++;
      continue;
    }

    if (DRY_RUN) {
      console.log(`  [dry-run] would migrate ${id}: status=${status}, turnNumber=${current.turnNumber}`);
      countMigrated++;
      continue;
    }

    // Accumulate into batch
    batchPayload[`games/${id}/current`] = current;
    batchCount++;
    countMigrated++;

    console.log(`  Queued ${id}: status=${status}, turnNumber=${current.turnNumber}`);

    if (batchCount >= BATCH_SIZE) {
      process.stdout.write(`  Flushing batch of ${batchCount} games … `);
      try {
        await flushBatch(batchPayload);
        console.log('done.');
      } catch (err) {
        console.error(`\n  ERROR flushing batch: ${err.message}`);
        countErrors += batchCount;
        countMigrated -= batchCount;
      }
      batchPayload = {};
      batchCount = 0;
    }
  }

  // Flush remainder
  if (batchCount > 0) {
    process.stdout.write(`  Flushing final batch of ${batchCount} games … `);
    try {
      await flushBatch(batchPayload);
      console.log('done.');
    } catch (err) {
      console.error(`\n  ERROR flushing final batch: ${err.message}`);
      countErrors += batchCount;
      countMigrated -= batchCount;
    }
  }

  // ─── Summary ───────────────────────────────────────────────────────────────
  console.log('\n─── Migration summary ───────────────────────────────────');
  console.log(`  Scanned : ${gameIds.length}`);
  console.log(`  Migrated: ${countMigrated}${DRY_RUN ? ' (dry-run, no writes)' : ''}`);
  console.log(`  Skipped : ${countSkipped}  (already had current, wrong status, or no moves)`);
  console.log(`  Errors  : ${countErrors}`);
  console.log('─────────────────────────────────────────────────────────');

  if (countErrors > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
