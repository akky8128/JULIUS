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
  let cpuSettings = null;

  if (settings.gameType === "offline") {
    players = {white: uid, black: uid};
    status = "in_progress";
  } else if (settings.gameType === "cpu") {
    const cpuLevel = Number(settings.cpuLevel);
    if (![1, 2, 3].includes(cpuLevel)) {
      throw new HttpsError("invalid-argument", "CPU level must be 1, 2, or 3.");
    }
    const playerColor = settings.playerColor;
    if (!["white", "black", "random"].includes(playerColor)) {
      throw new HttpsError("invalid-argument", "Invalid player color.");
    }
    const humanColor = playerColor === "random" ?
      (Math.random() < 0.5 ? "white" : "black") :
      playerColor;
    const cpuColor = humanColor === "white" ? "black" : "white";
    players = {white: uid, black: uid};
    status = "in_progress";
    cpuSettings = {level: cpuLevel, color: cpuColor};
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

  // 新スキーマ: meta / current / moves/{n} に分割して書き込む。
  // moves/0 はボードスナップショット付き（初期状態）。
  // current は最新確定状態の単一ノード（今後のトランザクション対象）。
  const initialTimers = {white: timeLimit, black: timeLimit};
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
        ...(cpuSettings ? {cpu: cpuSettings} : {}),
      },
      createdAt: now,
      updatedAt: now,
      expiresAt: status === "waiting" ? expiresAt : null,
    },
    current: {
      turnNumber: 0,
      currentPlayer: "white",
      board: initialBoard,
      summonCounts: {white: 0, black: 0},
      timers: initialTimers,
      timestamp: now,
      status,
    },
    moves: {
      0: {
        turnNumber: 0,
        currentPlayer: "white",
        board: initialBoard,
        summonCounts: {white: 0, black: 0},
        timers: initialTimers,
        timestamp: now,
      },
    },
  };

  const db = admin.database();
  await db.ref(`/games/${gameId}`).set(gameData);

  if (settings.gameType === "offline" || settings.gameType === "cpu") {
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
  // moves/0 のタイマー・タイムスタンプをリセット（既存の挙動を維持）
  updates[`/games/${gameId}/moves/0/timers/white`] = timeLimit;
  updates[`/games/${gameId}/moves/0/timers/black`] = timeLimit;
  updates[`/games/${gameId}/moves/0/timestamp`] = now;
  // current ノードもあわせてリセット
  updates[`/games/${gameId}/current/timers/white`] = timeLimit;
  updates[`/games/${gameId}/current/timers/black`] = timeLimit;
  updates[`/games/${gameId}/current/timestamp`] = now;
  updates[`/games/${gameId}/current/status`] = "in_progress";
  updates[`/users/${creatorUid}/games/${gameId}`] = now;
  updates[`/users/${joinerUid}/games/${gameId}`] = now;
  updates[`/users/${creatorUid}/profile/gamesPlayed`] = admin.database.ServerValue.increment(1);
  updates[`/users/${joinerUid}/profile/gamesPlayed`] = admin.database.ServerValue.increment(1);

  await db.ref().update(updates);

  return {success: true};
});


// --- (3) 手番の確定（盤面ロジックはすべてサーバー側で検証） ---

/**
 * current ノードに対してトランザクションを実行し、手番を確定させる。
 * legacyFallback=true の場合、旧スキーマゲームに対して合成した current で再試行する。
 *
 * @param {object} db Firebase Admin Database インスタンス
 * @param {string} gameId ゲームID
 * @param {string} uid 呼び出しユーザーの UID
 * @param {number} turnNumber クライアントが送信した手番番号
 * @param {Array}  actions アクション配列
 * @param {object} meta meta ノードの値（players, gameSettings）
 * @returns {{ txResult, validationError, outcome }}
 */
async function runSubmitMoveTransaction(db, gameId, uid, turnNumber, actions, meta) {
  const currentRef = db.ref(`/games/${gameId}/current`);
  let validationError = null;
  let outcome = null;

  const txResult = await currentRef.transaction((currentData) => {
    validationError = null;
    outcome = null;

    if (!currentData) {
      // 初回呼び出しはローカル未キャッシュのため必ずnullで呼ばれる。
      // ここでundefinedを返すとSDKがサーバーの実データで再試行せず即abortしてしまうため、
      // nullを返して突合・再試行に委ね、本当に存在しない場合はトランザクション完了後に判定する。
      return null;
    }

    if (currentData.status !== "in_progress") {
      validationError = new HttpsError("failed-precondition", "This game is not in progress.");
      return undefined;
    }

    if (currentData.turnNumber !== turnNumber) {
      validationError = new HttpsError("aborted", "The turn has already advanced. Please reload.");
      return undefined;
    }

    const moverColor = currentData.currentPlayer;
    const moverUid = meta.players[moverColor];
    if (uid !== moverUid) {
      validationError = new HttpsError("permission-denied", "It is not your turn.");
      return undefined;
    }

    const {boardSize, maxSummons, timeControl} = meta.gameSettings;
    const newTimers = computeTimersAfterTurn(currentData, timeControl, moverColor, Date.now());
    if (timeControl.enabled && newTimers[moverColor] <= 0) {
      validationError = new HttpsError("failed-precondition", "Time is up; this move can no longer be submitted.");
      return undefined;
    }

    let simulated;
    try {
      simulated = simulateTurn({
        board: normalizeBoard(currentData.board, boardSize),
        summonCounts: currentData.summonCounts,
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
    const newTurnNumber = turnNumber + 1;

    const hasWinner = checkWinCondition(
      {board: simulated.board, summonCounts: simulated.summonCounts, currentPlayer: newCurrentPlayer},
      {maxSummons, boardSize},
    );
    if (hasWinner) {
      outcome = {
        winner: moverColor,
        winReason: `${newCurrentPlayer === "white" ? "白" : "黒"}は有効な手を指せなくなりました。`,
        actions,
        newTimers,
        newTurnNumber,
        newCurrentPlayer,
        simulated,
      };
    } else {
      outcome = {
        winner: null,
        winReason: null,
        actions,
        newTimers,
        newTurnNumber,
        newCurrentPlayer,
        simulated,
      };
    }

    // current を新しい状態に更新する（盤面スナップショット付き）。
    return {
      turnNumber: newTurnNumber,
      currentPlayer: newCurrentPlayer,
      board: simulated.board,
      summonCounts: simulated.summonCounts,
      timers: newTimers,
      timestamp: Date.now(),
      status: hasWinner ? "completed" : "in_progress",
    };
  });

  return {txResult, validationError, outcome};
}

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

  // meta は対局参加後に不変（players, gameSettings）なので一度だけ読む。
  // status のバリデーションはトランザクション内の current.status で行う。
  const metaSnap = await db.ref(`/games/${gameId}/meta`).once("value");
  if (!metaSnap.exists()) {
    throw new HttpsError("not-found", "The specified game does not exist.");
  }
  const meta = metaSnap.val();

  let {txResult, validationError, outcome} = await runSubmitMoveTransaction(
    db, gameId, uid, turnNumber, actions, meta,
  );

  // current ノードが存在しない（旧スキーマのゲーム）場合のレガシーフォールバック。
  // moves 配列から current を合成して書き込み、トランザクションを1回再試行する。
  if (!validationError && txResult.committed && txResult.snapshot.val() === null) {
    const movesSnap = await db.ref(`/games/${gameId}/moves`).once("value");
    if (movesSnap.exists()) {
      const movesArr = toMovesArray(movesSnap.val());
      const lastMove = movesArr[movesArr.length - 1];
      const synthesized = {
        turnNumber: lastMove.turnNumber,
        currentPlayer: lastMove.currentPlayer,
        board: lastMove.board,
        summonCounts: lastMove.summonCounts,
        timers: lastMove.timers,
        timestamp: lastMove.timestamp,
        // meta.status をそのまま引き継ぐ（"in_progress" をハードコードすると
        // 完了済みの旧スキーマゲームに手を追加できてしまう）。
        status: meta.status,
      };
      const currentRef = db.ref(`/games/${gameId}/current`);
      // 並行ライターに上書きされないよう、null のときだけ書き込む。
      await currentRef.transaction((existing) => existing === null ? synthesized : undefined);
      // メイントランザクションを再試行する。
      ({txResult, validationError, outcome} = await runSubmitMoveTransaction(
        db, gameId, uid, turnNumber, actions, meta,
      ));
    } else {
      // moves も存在しない場合はゲームが本当に存在しない。
      throw new HttpsError("not-found", "The specified game does not exist.");
    }
  }

  if (validationError) {
    throw validationError;
  }
  if (!txResult.committed) {
    throw new HttpsError("aborted", "Failed to save the move. Please try again.");
  }

  // トランザクション確定後: moves/{n} の追記と meta の更新を一括書き込み。
  const {winner, winReason, newTimers, newTurnNumber, newCurrentPlayer} = outcome;
  const postUpdates = {};
  // moves/{n}: アクションのみ（盤面スナップショットなし）
  postUpdates[`/games/${gameId}/moves/${newTurnNumber}`] = {
    turnNumber: newTurnNumber,
    currentPlayer: newCurrentPlayer,
    actions,
    timers: newTimers,
    timestamp: Date.now(),
  };
  postUpdates[`/games/${gameId}/meta/updatedAt`] = Date.now();
  if (winner) {
    postUpdates[`/games/${gameId}/meta/status`] = "completed";
    postUpdates[`/games/${gameId}/meta/winner`] = winner;
    postUpdates[`/games/${gameId}/meta/winReason`] = winReason;
  }
  await db.ref().update(postUpdates);

  return {success: true, gameEnded: !!winner, winner: winner ?? null};
});


// --- (4) 時間切れの確定（サーバー時刻を基準に検証） ---

/**
 * current ノードに対して時間切れのトランザクションを実行する。
 *
 * @param {object} db Firebase Admin Database インスタンス
 * @param {string} gameId ゲームID
 * @param {string} uid 呼び出しユーザーの UID
 * @param {object} meta meta ノードの値
 * @returns {{ txResult, validationError, outcome }}
 */
async function runClaimTimeoutTransaction(db, gameId, uid, meta) {
  const currentRef = db.ref(`/games/${gameId}/current`);
  let validationError = null;
  let outcome = null;

  const txResult = await currentRef.transaction((currentData) => {
    validationError = null;
    outcome = null;

    if (!currentData) {
      // submitMove 同様、初回呼び出しの見せかけのnullでabortしないようにする。
      return null;
    }

    if (currentData.status !== "in_progress") {
      validationError = new HttpsError("failed-precondition", "This game is not in progress.");
      return undefined;
    }

    const players = meta.players;
    if (uid !== players.white && uid !== players.black) {
      validationError = new HttpsError("permission-denied", "You are not a participant in this game.");
      return undefined;
    }

    const timeControl = meta.gameSettings?.timeControl;
    if (!timeControl?.enabled) {
      validationError = new HttpsError("failed-precondition", "This game does not use time control.");
      return undefined;
    }

    const moverColor = currentData.currentPlayer;
    const newTimers = computeTimersAfterTurn(currentData, timeControl, moverColor, Date.now());

    if (newTimers[moverColor] > 0) {
      validationError = new HttpsError("failed-precondition", "Time has not run out yet.");
      return undefined;
    }

    const winner = moverColor === "white" ? "black" : "white";
    const winReason = `${moverColor === "white" ? "白" : "黒"}の時間切れです。`;
    outcome = {winner, winReason, newTimers};

    // 時間切れプレイヤーのタイマーを0に確定させて current を完了状態にする。
    return {
      ...currentData,
      timers: newTimers,
      status: "completed",
    };
  });

  return {txResult, validationError, outcome};
}

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

  // meta は不変フィールド（players, gameSettings）の読み取りに使う。
  const metaSnap = await db.ref(`/games/${gameId}/meta`).once("value");
  if (!metaSnap.exists()) {
    throw new HttpsError("not-found", "The specified game does not exist.");
  }
  const meta = metaSnap.val();

  let {txResult, validationError, outcome} = await runClaimTimeoutTransaction(
    db, gameId, uid, meta,
  );

  // レガシーフォールバック: current がない旧スキーマのゲームに対応する。
  if (!validationError && txResult.committed && txResult.snapshot.val() === null) {
    const movesSnap = await db.ref(`/games/${gameId}/moves`).once("value");
    if (movesSnap.exists()) {
      const movesArr = toMovesArray(movesSnap.val());
      const lastMove = movesArr[movesArr.length - 1];
      const synthesized = {
        turnNumber: lastMove.turnNumber,
        currentPlayer: lastMove.currentPlayer,
        board: lastMove.board,
        summonCounts: lastMove.summonCounts,
        timers: lastMove.timers,
        timestamp: lastMove.timestamp,
        // meta.status をそのまま引き継ぐ（"in_progress" をハードコードすると
        // 完了済みの旧スキーマゲームに手を追加できてしまう）。
        status: meta.status,
      };
      const currentRef = db.ref(`/games/${gameId}/current`);
      await currentRef.transaction((existing) => existing === null ? synthesized : undefined);
      ({txResult, validationError, outcome} = await runClaimTimeoutTransaction(
        db, gameId, uid, meta,
      ));
    } else {
      throw new HttpsError("not-found", "The specified game does not exist.");
    }
  }

  if (validationError) {
    throw validationError;
  }
  if (!txResult.committed) {
    throw new HttpsError("aborted", "Failed to finalize the timeout. Please try again.");
  }

  // トランザクション確定後: meta を一括更新する。
  const {winner, winReason} = outcome;
  const postUpdates = {};
  postUpdates[`/games/${gameId}/meta/status`] = "completed";
  postUpdates[`/games/${gameId}/meta/winner`] = winner;
  postUpdates[`/games/${gameId}/meta/winReason`] = winReason;
  postUpdates[`/games/${gameId}/meta/updatedAt`] = Date.now();
  await db.ref().update(postUpdates);

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
