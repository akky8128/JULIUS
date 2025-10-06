/**
 * Import function triggers from their respective submodules:
 *
 * const {onCall} = require("firebase-functions/v2/https");
 * const {onDocumentWritten} = require("firebase-functions/v2/firestore");
 *
 * See a full list of supported triggers at https://firebase.google.com/docs/functions
 */

const {onCall, HttpsError} = require("firebase-functions/v2/https");
const {onSchedule} = require("firebase-functions/v2/scheduler");
const {setGlobalOptions} = require("firebase-functions/v2");

const admin = require("firebase-admin");
const crypto = require("crypto");

admin.initializeApp();

setGlobalOptions({maxInstances: 10});

// --- (1) 対局作成関数 ---
exports.createGame = onCall({region: "asia-southeast1"}, async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }

  const uid = request.auth.uid;
  const settings = request.data;
  const now = admin.database.ServerValue.TIMESTAMP;
  const expiresAt = Date.now() + 5 * 60 * 1000; // 5分後に有効期限を設定
  const gameId = crypto.randomUUID();

  // --- 設定値のバリデーション ---
  const boardSize = parseInt(settings.boardSize, 10);
  if (boardSize < 3 || boardSize > 8) {
    throw new HttpsError(
        "invalid-argument",
        "Board size must be between 3 and 8.",
    );
  }
  const timeLimit = parseInt(settings.timeLimit, 10) * 60;
  const delay = parseInt(settings.delay, 10);
  const timeControlEnabled = settings["time-control-enabled"] === "on";

  let players;
  let status;

  if (settings.gameType === "offline") {
    players = {white: uid, black: uid};
    status = "in_progress";
  } else { // online
    const playerColor = settings.playerColor;
    // ▼▼▼【変更点】ランダムの場合、手番を決めずに作成者として記録 ▼▼▼
    if (playerColor === "random") {
      players = {creator: uid, white: 0, black: 0};
    } else {
      players = {
        creator: uid,
        white: playerColor === "white" ? uid : 0,
        black: playerColor === "black" ? uid : 0,
      };
    }
    status = "waiting";
  }

  const initialBoard = Array.from({length: boardSize}, () =>
    Array.from({length: boardSize}, () => 0),
  );

  const gameData = {
    gameId: gameId,
    meta: {
      status: status,
      players: players,
      winner: null,
      winReason: null,
      gameSettings: {
        boardSize,
        maxSummons: Math.floor(boardSize*boardSize/2),
        timeControl: {
          enabled: timeControlEnabled,
          timeLimit,
          delay,
        },
      },
      createdAt: now,
      updatedAt: now,
      // ▼▼▼【変更点】待機中のゲームにのみ有効期限を設定 ▼▼▼
      expiresAt: status === "waiting" ? expiresAt : null,
    },
    moves: [{
      turnNumber: 0,
      currentPlayer: "white",
      board: initialBoard,
      summonCounts: {white: 0, black: 0},
      timers: {white: timeLimit, black: timeLimit},
      timestamp: now,
    }],
  };

  const db = admin.database();
  await db.ref(`/games/${gameId}`).set(gameData);

  // オフラインゲームの場合のみ、この時点で作った人の対戦履歴を更新
  if (settings.gameType === "offline") {
    const userGamesRef = db.ref(`/users/${uid}/games/${gameId}`);
    await userGamesRef.set(now);
    const gamesPlayedRef = db.ref(`/users/${uid}/profile/gamesPlayed`);
    await gamesPlayedRef.set(admin.database.ServerValue.increment(1));
  }

  return {gameId: gameId};
});


// --- (2) 対局参加関数 ---
exports.joinGame = onCall({region: "asia-southeast1"}, async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }

  const joinerUid = request.auth.uid;
  const gameId = request.data.gameId;

  if (!gameId) {
    throw new HttpsError("invalid-argument", "gameId is required.");
  }

  const db = admin.database();
  const gameRef = db.ref(`/games/${gameId}`);
  const gameSnapshot = await gameRef.once("value");

  if (!gameSnapshot.exists()) {
    throw new HttpsError("not-found", "The specified game does not exist.");
  }

  const gameData = gameSnapshot.val();
  const {status, players} = gameData.meta;

  // --- バリデーション ---
  if (status !== "waiting") {
    throw new HttpsError(
        "failed-precondition",
        "This game is not waiting for players.",
    );
  }
  if (players.white !== 0 && players.black !== 0) {
    throw new HttpsError("failed-precondition", "This game is already full.");
  }
  if (
    players.creator ===
     joinerUid || players.white ===
      joinerUid || players.black ===
       joinerUid
  ) {
    throw new HttpsError(
        "failed-precondition",
        "You have already joined this game.",
    );
  }

  // --- 対局成立処理 ---
  const creatorUid = players.creator;
  let finalPlayers = {};

  // 手番が未決定の場合（ランダム）、ここで抽選する
  if (players.white === 0 && players.black === 0) {
    if (Math.random() < 0.5) {
      finalPlayers = {white: creatorUid, black: joinerUid};
    } else {
      finalPlayers = {white: joinerUid, black: creatorUid};
    }
  } else { // 手番が決まっている場合
    finalPlayers = {
      white: players.white === 0 ? joinerUid : players.white,
      black: players.black === 0 ? joinerUid : players.black,
    };
  }

  const now = admin.database.ServerValue.TIMESTAMP;
  const timeLimit = gameData.meta.gameSettings.timeControl.timeLimit;

  // 複数のパスを同時に更新
  const updates = {};
  updates[`/games/${gameId}/meta/status`] = "in_progress";
  updates[`/games/${gameId}/meta/players/white`] = finalPlayers.white;
  updates[`/games/${gameId}/meta/players/black`] = finalPlayers.black;
  updates[`/games/${gameId}/meta/players/creator`] = null;
  updates[`/games/${gameId}/meta/expiresAt`] = null;
  updates[`/games/${gameId}/meta/updatedAt`] = now;
  updates[`/games/${gameId}/moves/0/timers/white`] = timeLimit;
  updates[`/games/${gameId}/moves/0/timers/black`] = timeLimit;
  updates[`/games/${gameId}/moves/0/timestamp`] = now;
  // 両プレイヤーの対戦履歴を追加
  updates[`/users/${creatorUid}/games/${gameId}`] = now;
  updates[`/users/${joinerUid}/games/${gameId}`] = now;
  // 両プレイヤーの対戦数を1増やす
  updates[`/users/${creatorUid}/profile/gamesPlayed`] =
   admin.database.ServerValue.increment(1);
  updates[`/users/${joinerUid}/profile/gamesPlayed`] =
   admin.database.ServerValue.increment(1);

  await db.ref().update(updates);

  return {success: true};
});


// --- (3) ゴースト対局クリーンアップ関数 ---
exports.cleanupGhostGames = onSchedule({
  schedule: "every 5 minutes",
  region: "asia-southeast1",
},
async (event) => {
  const db = admin.database();
  const gamesRef = db.ref("/games");
  const now = Date.now();

  const query = gamesRef.orderByChild("meta/status").equalTo("waiting");
  const snapshot = await query.once("value");

  if (!snapshot.exists()) {
    console.log("No ghost games to clean up.");
    return null;
  }

  const updates = {};
  snapshot.forEach((childSnapshot) => {
    const game = childSnapshot.val();
    if (game.meta.expiresAt && game.meta.expiresAt < now) {
      console.log(`Deleting ghost game: ${childSnapshot.key}`);
      updates[childSnapshot.key] = null;
    }
  });

  if (Object.keys(updates).length > 0) {
    await gamesRef.update(updates);
  }

  return null;
});
