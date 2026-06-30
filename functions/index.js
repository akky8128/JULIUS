import {onCall, HttpsError} from "firebase-functions/v2/https";
import {setGlobalOptions} from "firebase-functions/v2";
import {onSchedule} from "firebase-functions/v2/scheduler";
import admin from "firebase-admin";
import crypto from "crypto";

admin.initializeApp({
  databaseURL: "https://julius-online-a5984-default-rtdb.asia-southeast1.firebasedatabase.app",
});

setGlobalOptions({maxInstances: 10, region: "asia-southeast1"});

// --- (1) 対局作成関数 ---
export const createGame = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }

  const uid = request.auth.uid;
  const settings = request.data;
  const now = admin.database.ServerValue.TIMESTAMP;
  const expiresAt = Date.now() + 5 * 60 * 1000;
  const gameId = crypto.randomUUID();

  const boardSize = parseInt(settings.boardSize, 10);
  if (boardSize < 3 || boardSize > 8) {
    throw new HttpsError("invalid-argument", "Board size must be between 3 and 8.");
  }
  const timeLimit = parseInt(settings.timeLimit, 10) * 60;
  const delay = parseInt(settings.delay, 10);
  const timeControlEnabled = settings["time-control-enabled"] === "on";

  let players;
  let status;

  if (settings.gameType === "offline") {
    players = {white: uid, black: uid};
    status = "in_progress";
  } else {
    const playerColor = settings.playerColor;
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
    gameId,
    meta: {
      status,
      players,
      winner: null,
      winReason: null,
      gameSettings: {
        boardSize,
        maxSummons: Math.floor(boardSize * boardSize / 2),
        timeControl: {enabled: timeControlEnabled, timeLimit, delay},
      },
      createdAt: now,
      updatedAt: now,
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

  if (settings.gameType === "offline") {
    const updates = {};
    updates[`/users/${uid}/games/${gameId}`] = now;
    updates[`/users/${uid}/profile/gamesPlayed`] = admin.database.ServerValue.increment(1);
    await db.ref().update(updates);
  }

  return {gameId};
});


// --- (2) 対局参加関数 ---
export const joinGame = onCall(async (request) => {
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

  if (status !== "waiting") {
    throw new HttpsError("failed-precondition", "This game is not waiting for players.");
  }
  if (players.white !== 0 && players.black !== 0) {
    throw new HttpsError("failed-precondition", "This game is already full.");
  }
  if (players.creator === joinerUid || players.white === joinerUid || players.black === joinerUid) {
    throw new HttpsError("failed-precondition", "You have already joined this game.");
  }

  const creatorUid = players.creator;
  let finalPlayers;

  if (players.white === 0 && players.black === 0) {
    finalPlayers = Math.random() < 0.5 ?
      {white: creatorUid, black: joinerUid} :
      {white: joinerUid, black: creatorUid};
  } else {
    finalPlayers = {
      white: players.white === 0 ? joinerUid : players.white,
      black: players.black === 0 ? joinerUid : players.black,
    };
  }

  const now = admin.database.ServerValue.TIMESTAMP;
  const timeLimit = gameData.meta.gameSettings.timeControl.timeLimit;

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
  updates[`/users/${creatorUid}/games/${gameId}`] = now;
  updates[`/users/${joinerUid}/games/${gameId}`] = now;
  updates[`/users/${creatorUid}/profile/gamesPlayed`] = admin.database.ServerValue.increment(1);
  updates[`/users/${joinerUid}/profile/gamesPlayed`] = admin.database.ServerValue.increment(1);

  await db.ref().update(updates);

  return {success: true};
});


// --- (3) ゴースト対局クリーンアップ ---
const db = admin.database();

export const cleanupGhostGames = onSchedule("every day 03:00", async () => {
  const gamesRef = db.ref("/games");
  const now = Date.now();
  const fiveMinutesAgo = now - 5 * 60 * 1000;

  const updates = {};

  const [abandonedSnapshot, waitingSnapshot] = await Promise.all([
    gamesRef.orderByChild("meta/status").equalTo("abandoned").once("value"),
    gamesRef.orderByChild("meta/status").equalTo("waiting").once("value"),
  ]);

  if (abandonedSnapshot.exists()) {
    abandonedSnapshot.forEach((gameSnap) => {
      updates[gameSnap.key] = null;
    });
  }

  if (waitingSnapshot.exists()) {
    waitingSnapshot.forEach((gameSnap) => {
      const gameData = gameSnap.val();
      if (gameData.meta?.createdAt && gameData.meta.createdAt < fiveMinutesAgo) {
        updates[gameSnap.key] = null;
      }
    });
  }

  if (Object.keys(updates).length > 0) {
    await gamesRef.update(updates);
    console.log(`Deleted ${Object.keys(updates).length} ghost games.`);
  }

  return null;
});
