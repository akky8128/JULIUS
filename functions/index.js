import {onCall, HttpsError} from "firebase-functions/v2/https";
import {setGlobalOptions} from "firebase-functions/v2";
import {onSchedule} from "firebase-functions/v2/scheduler";
import admin from "firebase-admin";
import crypto from "crypto";
import {simulateTurn, checkWinCondition, normalizeBoard, computeTimersAfterTurn, GameLogicError} from "./gameLogic.js";

const MAX_ACTIONS_PER_TURN = 64;
const MAX_NICKNAME_LENGTH = 50;
const MAX_BIO_LENGTH = 300;

function toMovesArray(moves) {
  if (!moves) return [];
  return Array.isArray(moves) ? moves : Object.values(moves);
}

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

  const boardSize = Number(settings.boardSize);
  if (!Number.isInteger(boardSize) || boardSize < 3 || boardSize > 8) {
    throw new HttpsError("invalid-argument", "Board size must be an integer between 3 and 8.");
  }

  const timeControlEnabled = settings["time-control-enabled"] === "on";
  let timeLimitMinutes = 0;
  let delay = 0;
  if (timeControlEnabled) {
    timeLimitMinutes = Number(settings.timeLimit);
    if (!Number.isInteger(timeLimitMinutes) || timeLimitMinutes < 1 || timeLimitMinutes > 180) {
      throw new HttpsError("invalid-argument", "Time limit must be an integer between 1 and 180 minutes.");
    }
    delay = Number(settings.delay);
    if (!Number.isInteger(delay) || delay < 0 || delay > 60) {
      throw new HttpsError("invalid-argument", "Delay must be an integer between 0 and 60 seconds.");
    }
  }
  const timeLimit = timeLimitMinutes * 60;

  let players;
  let status;

  if (settings.gameType === "offline") {
    players = {white: uid, black: uid};
    status = "in_progress";
  } else if (settings.gameType === "online") {
    const playerColor = settings.playerColor;
    if (!["white", "black", "random"].includes(playerColor)) {
      throw new HttpsError("invalid-argument", "Invalid player color.");
    }
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
  } else {
    throw new HttpsError("invalid-argument", "Invalid game type.");
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


// --- (3) 手番の確定（盤面ロジックはすべてサーバー側で検証） ---
export const submitMove = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }

  const uid = request.auth.uid;
  const {gameId, turnNumber, actions} = request.data || {};

  if (!gameId || typeof gameId !== "string") {
    throw new HttpsError("invalid-argument", "gameId is required.");
  }
  if (!Number.isInteger(turnNumber) || turnNumber < 0) {
    throw new HttpsError("invalid-argument", "turnNumber is required.");
  }
  if (!Array.isArray(actions) || actions.length === 0 || actions.length > MAX_ACTIONS_PER_TURN) {
    throw new HttpsError("invalid-argument", "actions must be a non-empty array.");
  }

  const db = admin.database();
  const gameRef = db.ref(`/games/${gameId}`);

  let validationError = null;
  let outcome = null;

  const txResult = await gameRef.transaction((gameData) => {
    validationError = null;
    outcome = null;

    if (!gameData) {
      // 初回呼び出しはローカル未キャッシュのため必ずnullで呼ばれる。
      // ここでundefinedを返すとSDKがサーバーの実データで再試行せず即abortしてしまうため、
      // nullを返して突合・再試行に委ね、本当に存在しない場合はトランザクション完了後に判定する。
      return null;
    }
    if (gameData.meta?.status !== "in_progress") {
      validationError = new HttpsError("failed-precondition", "This game is not in progress.");
      return undefined;
    }

    const moves = toMovesArray(gameData.moves);
    if (moves.length === 0) {
      validationError = new HttpsError("failed-precondition", "Game has no move history.");
      return undefined;
    }
    const lastMove = moves[moves.length - 1];

    if (lastMove.turnNumber !== turnNumber) {
      validationError = new HttpsError("aborted", "The turn has already advanced. Please reload.");
      return undefined;
    }

    const moverColor = lastMove.currentPlayer;
    const moverUid = gameData.meta.players[moverColor];
    if (uid !== moverUid) {
      validationError = new HttpsError("permission-denied", "It is not your turn.");
      return undefined;
    }

    const {boardSize, maxSummons, timeControl} = gameData.meta.gameSettings;
    const newTimers = computeTimersAfterTurn(lastMove, timeControl, moverColor, Date.now());
    if (timeControl.enabled && newTimers[moverColor] <= 0) {
      validationError = new HttpsError("failed-precondition", "Time is up; this move can no longer be submitted.");
      return undefined;
    }

    let simulated;
    try {
      simulated = simulateTurn({
        board: normalizeBoard(lastMove.board, boardSize),
        summonCounts: lastMove.summonCounts,
        currentPlayer: moverColor,
        boardSize,
      }, actions);
    } catch (err) {
      validationError = err instanceof GameLogicError ?
        new HttpsError(err.code, err.message) :
        new HttpsError("internal", "Failed to process the move.");
      return undefined;
    }

    const newCurrentPlayer = moverColor === "white" ? "black" : "white";
    const newMove = {
      turnNumber: lastMove.turnNumber + 1,
      currentPlayer: newCurrentPlayer,
      board: simulated.board,
      summonCounts: simulated.summonCounts,
      timers: newTimers,
      timestamp: Date.now(),
    };
    moves.push(newMove);

    const updatedMeta = {...gameData.meta, updatedAt: Date.now()};

    const hasWinner = checkWinCondition(
      {board: simulated.board, summonCounts: simulated.summonCounts, currentPlayer: newCurrentPlayer},
      {maxSummons, boardSize},
    );
    if (hasWinner) {
      updatedMeta.status = "completed";
      updatedMeta.winner = moverColor;
      updatedMeta.winReason = `${newCurrentPlayer === "white" ? "白" : "黒"}は有効な手を指せなくなりました。`;
      outcome = {winner: moverColor};
    }

    return {...gameData, meta: updatedMeta, moves};
  });

  if (!validationError && txResult.committed && txResult.snapshot.val() === null) {
    validationError = new HttpsError("not-found", "The specified game does not exist.");
  }
  if (validationError) {
    throw validationError;
  }
  if (!txResult.committed) {
    throw new HttpsError("aborted", "Failed to save the move. Please try again.");
  }

  return {success: true, gameEnded: !!outcome, winner: outcome ? outcome.winner : null};
});


// --- (4) 時間切れの確定（サーバー時刻を基準に検証） ---
export const claimTimeout = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }

  const uid = request.auth.uid;
  const {gameId} = request.data || {};
  if (!gameId || typeof gameId !== "string") {
    throw new HttpsError("invalid-argument", "gameId is required.");
  }

  const db = admin.database();
  const gameRef = db.ref(`/games/${gameId}`);

  let validationError = null;
  let outcome = null;

  const txResult = await gameRef.transaction((gameData) => {
    validationError = null;
    outcome = null;

    if (!gameData) {
      // submitMove同様、初回呼び出しの見せかけのnullでabortしないようにする。
      return null;
    }
    if (gameData.meta?.status !== "in_progress") {
      validationError = new HttpsError("failed-precondition", "This game is not in progress.");
      return undefined;
    }
    const players = gameData.meta.players;
    if (uid !== players.white && uid !== players.black) {
      validationError = new HttpsError("permission-denied", "You are not a participant in this game.");
      return undefined;
    }
    const timeControl = gameData.meta.gameSettings?.timeControl;
    if (!timeControl?.enabled) {
      validationError = new HttpsError("failed-precondition", "This game does not use time control.");
      return undefined;
    }

    const moves = toMovesArray(gameData.moves);
    if (moves.length === 0) {
      validationError = new HttpsError("failed-precondition", "Game has no move history.");
      return undefined;
    }
    const lastMove = moves[moves.length - 1];
    const moverColor = lastMove.currentPlayer;
    const newTimers = computeTimersAfterTurn(lastMove, timeControl, moverColor, Date.now());

    if (newTimers[moverColor] > 0) {
      validationError = new HttpsError("failed-precondition", "Time has not run out yet.");
      return undefined;
    }

    const winner = moverColor === "white" ? "black" : "white";
    const winReason = `${moverColor === "white" ? "白" : "黒"}の時間切れです。`;
    outcome = {winner};

    moves[moves.length - 1] = {...lastMove, timers: newTimers};

    return {
      ...gameData,
      meta: {...gameData.meta, status: "completed", winner, winReason, updatedAt: Date.now()},
      moves,
    };
  });

  if (!validationError && txResult.committed && txResult.snapshot.val() === null) {
    validationError = new HttpsError("not-found", "The specified game does not exist.");
  }
  if (validationError) {
    throw validationError;
  }
  if (!txResult.committed) {
    throw new HttpsError("aborted", "Failed to finalize the timeout. Please try again.");
  }

  return {success: true, winner: outcome.winner};
});


// --- (5) フォロー / アンフォロー ---
export const toggleFollow = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }
  const uid = request.auth.uid;
  const targetUid = request.data?.targetUid;

  if (!targetUid || typeof targetUid !== "string") {
    throw new HttpsError("invalid-argument", "targetUid is required.");
  }
  if (targetUid === uid) {
    throw new HttpsError("invalid-argument", "You cannot follow yourself.");
  }

  const db = admin.database();
  const followingRef = db.ref(`/users/${uid}/following/${targetUid}`);

  // トランザクションでトグルの向きを確定させ、二重クリックによる競合を防ぐ。
  const txResult = await followingRef.transaction((current) => (current ? null : true));
  const isNowFollowing = txResult.snapshot.val() === true;

  const updates = {};
  updates[`/users/${targetUid}/followers/${uid}`] = isNowFollowing ? true : null;
  updates[`/users/${uid}/profile/followingCount`] = admin.database.ServerValue.increment(isNowFollowing ? 1 : -1);
  updates[`/users/${targetUid}/profile/followersCount`] = admin.database.ServerValue.increment(isNowFollowing ? 1 : -1);

  if (isNowFollowing) {
    const followerNicknameSnap = await db.ref(`/users/${uid}/profile/nickname`).once("value");
    const followerNickname = followerNicknameSnap.exists() ? followerNicknameSnap.val() : "ゲスト";
    updates[`/followNotifications/${targetUid}/${uid}`] = {
      followerNickname,
      timestamp: admin.database.ServerValue.TIMESTAMP,
    };
  } else {
    updates[`/followNotifications/${targetUid}/${uid}`] = null;
  }

  await db.ref().update(updates);

  return {following: isNowFollowing};
});


// --- (6) プロフィール編集 ---
export const updateProfile = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }
  const uid = request.auth.uid;
  const {nickname, bio} = request.data || {};

  if (typeof nickname !== "string" || nickname.trim().length === 0) {
    throw new HttpsError("invalid-argument", "Nickname is required.");
  }
  const trimmedNickname = nickname.trim();
  if (trimmedNickname.length > MAX_NICKNAME_LENGTH) {
    throw new HttpsError("invalid-argument", `Nickname must be ${MAX_NICKNAME_LENGTH} characters or fewer.`);
  }
  if (typeof bio !== "string") {
    throw new HttpsError("invalid-argument", "Bio must be a string.");
  }
  const trimmedBio = bio.trim();
  if (trimmedBio.length > MAX_BIO_LENGTH) {
    throw new HttpsError("invalid-argument", `Bio must be ${MAX_BIO_LENGTH} characters or fewer.`);
  }

  const db = admin.database();
  const profileRef = db.ref(`/users/${uid}/profile`);
  const snapshot = await profileRef.once("value");
  const currentProfile = snapshot.exists() ? snapshot.val() : {};

  const updatedProfile = {
    ...currentProfile,
    nickname: trimmedNickname,
    bio: trimmedBio,
    gamesPlayed: currentProfile.gamesPlayed || 0,
    followingCount: currentProfile.followingCount || 0,
    followersCount: currentProfile.followersCount || 0,
  };

  await profileRef.set(updatedProfile);
  return {success: true};
});


// --- (7) 対局招待の作成 ---
export const createInvitation = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }
  const uid = request.auth.uid;
  const {gameId, targetUid} = request.data || {};

  if (!gameId || typeof gameId !== "string") {
    throw new HttpsError("invalid-argument", "gameId is required.");
  }
  if (!targetUid || typeof targetUid !== "string") {
    throw new HttpsError("invalid-argument", "targetUid is required.");
  }
  if (targetUid === uid) {
    throw new HttpsError("invalid-argument", "You cannot invite yourself.");
  }

  const db = admin.database();
  const metaSnap = await db.ref(`/games/${gameId}/meta`).once("value");
  if (!metaSnap.exists()) {
    throw new HttpsError("not-found", "The specified game does not exist.");
  }
  const meta = metaSnap.val();
  if (meta.status !== "waiting") {
    throw new HttpsError("failed-precondition", "This game is no longer waiting for players.");
  }
  if (meta.players?.creator !== uid) {
    throw new HttpsError("permission-denied", "Only the game creator can send invitations.");
  }

  const inviterNicknameSnap = await db.ref(`/users/${uid}/profile/nickname`).once("value");
  const inviterNickname = inviterNicknameSnap.exists() ? inviterNicknameSnap.val() : "ゲスト";

  await db.ref(`/invitations/${targetUid}/${gameId}`).set({
    inviterUid: uid,
    inviterNickname,
    timestamp: admin.database.ServerValue.TIMESTAMP,
  });

  return {success: true};
});


// --- (8) 招待の削除（期限切れ・無効化された招待の自己クリーンアップ含む） ---
export const dismissInvitation = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }
  const uid = request.auth.uid;
  const {gameId} = request.data || {};

  if (!gameId || typeof gameId !== "string") {
    throw new HttpsError("invalid-argument", "gameId is required.");
  }

  await admin.database().ref(`/invitations/${uid}/${gameId}`).remove();
  return {success: true};
});


// --- (9) フォロー通知の既読化 ---
export const dismissFollowNotification = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Authentication is required.");
  }
  const uid = request.auth.uid;
  const {followerUid} = request.data || {};

  if (!followerUid || typeof followerUid !== "string") {
    throw new HttpsError("invalid-argument", "followerUid is required.");
  }

  await admin.database().ref(`/followNotifications/${uid}/${followerUid}`).remove();
  return {success: true};
});


// --- (10) ゴースト対局クリーンアップ ---
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
