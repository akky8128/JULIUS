// ===================================================================
// チュートリアルのステップ進行エンジン。
// ルール判定は一切再実装せず、js/gameLogic.js の simulateTurn に委譲する。
//
// Node でテストできるように、DOM に依存する部分（盤面描画・クリックハンドラの登録）と
// 純粋なロジック部分（アクション組み立て・期待値照合）を分離している。
// 純粋関数群は下の「純粋ロジック」セクションにまとめてあり、TutorialEngine クラスから
// そのまま呼び出す。
//
// 主な仕様:
//  - info ステップ（解説のみ、盤面クリックでは進まず next() で進む）
//  - tap ステップ（expect.type==='tap'：座標を覚えるための「タップするだけ」の練習。
//    正解しても自動遷移せず「次へ」で進む）
//  - 召喚 / 排除 / 移動 は「ターン終了」を押すまで確定しない（自動遷移しない）。
//    これにより連続移動が可能になり、また操作結果を確認する時間ができる。
//  - legalDestinations / maxMovableCount（game.html の移動UXと同ロジックの純粋関数）
//  - 誤答が一定回数に達したらヒント（ハイライト）を出す（revealHighlightAfterWrong）
//  - 枚数選択コールバック（promptCount）と onSuccess コールバック
// ===================================================================

import { simulateTurn, GameLogicError, cloneBoard, denormalizeBoard, normalizeStack } from "../gameLogic.js";
import { renderBoard } from "./boardView.js";
import { parseCoord, diffToNotation } from "./coords.js";

const STORAGE_KEY = "julius-tutorial-progress";

// ───────────────────────── 純粋ロジック（Nodeでテスト可能） ─────────────────────────

/**
 * 指定マスから移動できる先（直交隣接の在盤4マス）を返す。
 * game.html の getValidMoves と同一ロジック。
 *
 * @param {number} boardSize
 * @param {{r:number,c:number}} from
 * @returns {Array<{r:number,c:number}>}
 */
export function legalDestinations(boardSize, from) {
  const directions = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];
  const moves = [];
  for (const [dr, dc] of directions) {
    const newR = from.r + dr;
    const newC = from.c + dc;
    if (newR >= 0 && newR < boardSize && newC >= 0 && newC < boardSize) {
      moves.push({ r: newR, c: newC });
    }
  }
  return moves;
}

/**
 * 指定マスから動かせる最大枚数を返す（最上段に連続する自駒の枚数）。
 * 連続移動中（isContinuing=true）は自駒を1枚残す必要があるため -1 する。
 * game.html の promptForPieceCount の枚数計算ロジックと同一。
 *
 * @param {Array} board - denormalized board
 * @param {{r:number,c:number}} from
 * @param {string} player - 'white' | 'black'
 * @param {boolean} isContinuing
 * @returns {number} 0以下なら移動不可
 */
export function maxMovableCount(board, from, player, isContinuing) {
  const stack = normalizeStack(board?.[from.r]?.[from.c]);
  let movablePiecesCount = 0;
  for (let i = stack.length - 1; i >= 0; i--) {
    if (stack[i] === player) movablePiecesCount++;
    else break;
  }
  return isContinuing ? movablePiecesCount - 1 : movablePiecesCount;
}

/**
 * シナリオの expect 定義と、実際に組んだ actions を照合する。
 *
 * @param {Array} actions - simulateTurn に渡したアクション配列。
 * @param {object} expect - シナリオの期待値定義。
 *   - {type:'summon', at:'c3'}
 *   - {type:'eliminate', at:'c3'}
 *   - {type:'move', from:'b2', to:'b3'} … 連続移動の場合は「最初のfrom」と「最後のto」で照合
 *   - {type:'moveChain', steps:[{from,to},...]} … 連続移動の各ステップを厳密に照合
 *   - {type:'any'} … 合法手であれば何でもよい
 *   （'tap' は盤面操作を伴わないため、ここでは扱わずエンジン側で座標一致を見る）
 * @param {number} boardSize
 * @returns {boolean}
 */
export function matchesExpectation(actions, expect, boardSize) {
  if (!expect || expect.type === "any") {
    // simulateTurn が例外を投げずに通った時点で合法手であることは保証されているため、
    // ここでは常に true でよい（呼び出し側で simulateTurn の成否を先に確認すること）。
    return true;
  }

  if (!Array.isArray(actions) || actions.length === 0) return false;

  if (expect.type === "summon") {
    if (actions.length !== 1 || actions[0].type !== "summon") return false;
    const target = parseCoord(expect.at, boardSize);
    return actions[0].r === target.r && actions[0].c === target.c;
  }

  if (expect.type === "eliminate") {
    if (actions.length !== 1 || actions[0].type !== "eliminate") return false;
    const target = parseCoord(expect.at, boardSize);
    return actions[0].r === target.r && actions[0].c === target.c;
  }

  if (expect.type === "move") {
    // 連続移動も含め、「最初のfrom」と「最後のto」が一致すればよい。
    if (actions.length === 0 || actions.some((a) => a.type !== "move")) return false;
    const from = parseCoord(expect.from, boardSize);
    const to = parseCoord(expect.to, boardSize);
    const firstFrom = actions[0].from;
    const lastTo = actions[actions.length - 1].to;
    return (
      firstFrom.r === from.r &&
      firstFrom.c === from.c &&
      lastTo.r === to.r &&
      lastTo.c === to.c
    );
  }

  if (expect.type === "moveChain") {
    // 連続移動の各ステップ（from/to）を厳密に照合する。枚数は問わない。
    if (!Array.isArray(expect.steps) || expect.steps.length === 0) return false;
    if (actions.length !== expect.steps.length) return false;
    if (actions.some((a) => a.type !== "move")) return false;
    return expect.steps.every((step, i) => {
      const from = parseCoord(step.from, boardSize);
      const to = parseCoord(step.to, boardSize);
      const action = actions[i];
      return (
        action.from.r === from.r &&
        action.from.c === from.c &&
        action.to.r === to.r &&
        action.to.c === to.c
      );
    });
  }

  return false;
}

/**
 * summon アクションを組み立てる。
 */
export function buildSummonAction(r, c) {
  return { type: "summon", r, c };
}

/**
 * eliminate アクションを組み立てる。
 */
export function buildEliminateAction(r, c) {
  return { type: "eliminate", r, c };
}

/**
 * move アクションを組み立てる（1手分）。連続移動は呼び出し側で配列に連結する。
 */
export function buildMoveAction(from, to, count) {
  return { type: "move", from, to, count };
}

/**
 * シナリオの初期状態 + 組み立てたアクションを simulateTurn に渡して検証し、
 * 成功すれば {ok:true, board, summonCounts, notation} を、
 * 失敗すれば {ok:false, error} を返す純粋関数。
 */
export function tryApplyActions(scenario, actions) {
  const before = denormalizeBoard(cloneBoard(scenario.board));
  try {
    const result = simulateTurn(
      {
        board: scenario.board,
        summonCounts: scenario.summonCounts,
        currentPlayer: scenario.currentPlayer,
        boardSize: scenario.boardSize,
      },
      actions
    );
    const notation = diffToNotation(before, result.board, scenario.boardSize);
    return { ok: true, board: result.board, summonCounts: result.summonCounts, notation };
  } catch (err) {
    if (err instanceof GameLogicError) {
      return { ok: false, error: err.message };
    }
    throw err;
  }
}

// ───────────────────────── DOM依存のステップ進行エンジン ─────────────────────────

export class TutorialEngine {
  /**
   * @param {object} params
   * @param {HTMLElement} params.boardEl - 盤面を描画するコンテナ要素。
   * @param {Array} params.scenarios - scenarios.js の配列。
   * @param {(step:number, total:number) => void} [params.onProgress] - 進捗表示コールバック。
   * @param {(scenario:object) => void} [params.onMessage] - ガイドメッセージ表示コールバック（infoステップでも呼ばれる）。
   * @param {(notation:string) => void} [params.onKifu] - 棋譜欄への追記コールバック。
   * @param {(message:string) => void} [params.onWrong] - 不正解時のメッセージ表示コールバック。
   * @param {(message:string) => void} [params.onSuccess] - 正解・ヒント時のメッセージ表示コールバック。
   * @param {(max:number) => Promise<number|null>} [params.promptCount] - 移動枚数を尋ねるコールバック。
   *   未指定の場合は常に最大枚数を使うフォールバックになる。
   */
  constructor({ boardEl, scenarios, onProgress, onMessage, onKifu, onWrong, onSuccess, promptCount }) {
    this.boardEl = boardEl;
    this.scenarios = scenarios;
    this.onProgress = onProgress || (() => {});
    this.onMessage = onMessage || (() => {});
    this.onKifu = onKifu || (() => {});
    this.onWrong = onWrong || (() => {});
    this.onSuccess = onSuccess || (() => {});
    this.promptCount = promptCount || (async (max) => max);

    this.stepIndex = this._loadProgress();
    this.selected = null; // {r,c} 選択中のマス（移動元）
    this.pendingActions = [];
    // 連続移動中に一時反映している盤面（表示専用、確定はしていない）。
    this._workingBoard = null;
    // 'move' | 'eliminate'。自分の駒が一番上にあるマスをクリックした際に
    // 「移動元として選ぶ」のか「排除を試みる」のかを明示的に切り替える。
    this.actionMode = "move";
    // 誤答回数。ヒント（ハイライト）表示の閾値判定に使う。ステップが変わるとリセット。
    this.wrongCount = 0;
    // 座標タップ専用ステップ（expect.type==='tap'）で正解タップ済みかどうか。
    this.tapCleared = false;

    this._syncActionMode();
  }

  /**
   * 移動/排除の操作モードを切り替える。排除は召喚フェーズ終了後のみ有効な章で使う。
   * @param {'move'|'eliminate'} mode
   */
  setActionMode(mode) {
    if (mode !== "move" && mode !== "eliminate") return;
    this.actionMode = mode;
    this.selected = null;
    this._renderCurrentStep();
  }

  _loadProgress() {
    try {
      const raw = typeof localStorage !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
      const parsed = Number(raw);
      if (Number.isInteger(parsed) && parsed >= 0 && parsed < this.scenarios.length) {
        return parsed;
      }
    } catch {
      // localStorage が使えない環境（プライベートモード等）は無視して先頭から始める。
    }
    return 0;
  }

  _saveProgress() {
    try {
      if (typeof localStorage !== "undefined") {
        localStorage.setItem(STORAGE_KEY, String(this.stepIndex));
      }
    } catch {
      // 保存に失敗しても致命的ではないため無視する。
    }
  }

  get currentScenario() {
    return this.scenarios[this.stepIndex];
  }

  /** 現在のステップが解説専用（info）かどうか。 */
  get isInfoStep() {
    return (this.currentScenario?.kind || "action") === "info";
  }

  /** 現在のステップが座標タップ専用（tap）かどうか。 */
  get isTapStep() {
    return this.currentScenario?.expect?.type === "tap";
  }

  /** 「次へ」ボタンを出してよいか（解説ステップ、またはタップ正解済み）。 */
  get canProceed() {
    // 最終ステップ（締め）は「次へ」ではなく完了パネルを出すため、進めない扱いにする。
    if (this.isFinalStep) return false;
    return this.isInfoStep || this.tapCleared;
  }

  /** 現在のステップが最後（締め＝チュートリアル完了画面）かどうか。 */
  get isFinalStep() {
    return this.stepIndex === this.scenarios.length - 1;
  }

  /** 排除モードへの切り替えボタンをUIに出してよいか（許可された章かつ両者が召喚完了）。 */
  get canToggleEliminateMode() {
    const scenario = this.currentScenario;
    if (!scenario || !scenario.allowEliminateMode) return false;
    const { summonCounts, boardSize } = scenario;
    const maxSummons = Math.floor((boardSize * boardSize) / 2);
    return summonCounts.white === maxSummons && summonCounts.black === maxSummons;
  }

  /** pendingActions が1つ以上あり「ターン終了」を押せる状態かどうか。 */
  get canEndTurn() {
    return this.pendingActions.length > 0;
  }

  /** 表示に使う盤面（移動・召喚などを一時反映した盤面、なければシナリオの初期盤面）。 */
  get displayBoard() {
    return this._workingBoard || this.currentScenario.board;
  }

  /** ステップに応じて既定の操作モード（move / eliminate）を設定する。 */
  _syncActionMode() {
    this.actionMode = this.currentScenario?.defaultActionMode || "move";
  }

  /** ステップ内の一時状態（選択・保留アクション・タップ状態など）をリセットする。 */
  _resetStepState() {
    this.selected = null;
    this.pendingActions = [];
    this._workingBoard = null;
    this.tapCleared = false;
    this._syncActionMode();
  }

  start() {
    this.wrongCount = 0;
    this._resetStepState();
    this._renderCurrentStep();
  }

  restart() {
    // 現在ステップのやり直し。誤答回数は保持する（一度出したヒントは出したまま）。
    this._resetStepState();
    this._renderCurrentStep();
  }

  cancel() {
    this.restart();
  }

  /** info / tap ステップから次のステップへ進む。 */
  next() {
    this.advance();
  }

  /** このステップで実際に表示するハイライト。誤答が閾値未満なら隠す。 */
  _visibleHighlights() {
    const scenario = this.currentScenario;
    const highlights = scenario.highlights || [];
    const threshold = scenario.revealHighlightAfterWrong;
    if (typeof threshold === "number" && this.wrongCount < threshold) {
      return [];
    }
    return highlights;
  }

  /** 誤答時の共通処理。誤答回数を増やし、保留状態をリセットして再描画する。 */
  _registerWrong(message) {
    this.wrongCount += 1;
    this.pendingActions = [];
    this._workingBoard = null;
    this.selected = null;
    this._renderCurrentStep();
    this.onWrong(message);
  }

  /** 保留中のアクションがある間、次に何をすればよいかのヒントを出す。 */
  _hintAfterAction() {
    if (this.selected) {
      this.onSuccess("続けて動かせます。よければ「ターン終了」で確定しましょう。");
    } else {
      this.onSuccess("「ターン終了」ボタンを押して手を確定しましょう。");
    }
  }

  _renderCurrentStep() {
    const scenario = this.currentScenario;
    this.onProgress(this.stepIndex + 1, this.scenarios.length);
    this.onMessage(scenario);

    if (this.isInfoStep) {
      renderBoard(this.boardEl, scenario.board, scenario.boardSize, {
        selected: null,
        validMoves: [],
        highlights: scenario.highlights || [],
        onCellClick: null,
      });
      return;
    }

    const validMoves = this.selected
      ? legalDestinations(scenario.boardSize, this.selected)
      : [];

    renderBoard(this.boardEl, this.displayBoard, scenario.boardSize, {
      selected: this.selected,
      validMoves,
      highlights: this._visibleHighlights(),
      onCellClick: (r, c) => this._handleCellClick(r, c),
    });
  }

  _handleCellClick(r, c) {
    if (this.isInfoStep) return; // 解説ステップでは盤面クリックは無効。
    if (this.tapCleared) return; // タップ正解後は追加操作を受け付けない。

    const scenario = this.currentScenario;

    // 座標タップ専用ステップ: simulateTurn を通さず、座標の一致だけを判定する。
    if (this.isTapStep) {
      this._handleTap(scenario, r, c);
      return;
    }

    const board = this.displayBoard;
    const stack = normalizeStack(board?.[r]?.[c]);
    const isContinuing = this.pendingActions.length > 0;

    if (!this.selected) {
      if (isContinuing) {
        // 連続移動中は「直前の移動先」（自動選択される）からしか動かせない。
        return;
      }

      if (this.actionMode === "eliminate") {
        // 排除モード: 合否判定は tryApplyActions（＝simulateTurn）に委譲する。
        this._tryAddSingle(scenario, buildEliminateAction(r, c));
        return;
      }

      if (stack.length === 0) {
        // 空きマス: 召喚を試みる。
        this._tryAddSingle(scenario, buildSummonAction(r, c));
        return;
      }
      if (stack[stack.length - 1] === scenario.currentPlayer) {
        // 自分の駒: 移動元として選択。
        this.selected = { r, c };
        this._renderCurrentStep();
        return;
      }
      // 相手駒が一番上のマスは、移動モードでは操作できない（何もしない）。
      return;
    }

    // 移動先クリック。
    const from = this.selected;
    if (from.r === r && from.c === c) {
      this.selected = null;
      this._renderCurrentStep();
      return;
    }

    const destinations = legalDestinations(scenario.boardSize, from);
    if (!destinations.some((m) => m.r === r && m.c === c)) {
      // 隣接していない移動先は無視する。
      return;
    }

    this._beginOrContinueMove(scenario, from, { r, c });
  }

  /** 座標タップ専用ステップの判定。正解なら「次へ」待ちにする。 */
  _handleTap(scenario, r, c) {
    const target = parseCoord(scenario.expect.at, scenario.boardSize);
    if (r === target.r && c === target.c) {
      this.tapCleared = true;
      this.selected = { r, c }; // タップしたマスを選択表示する。
      this._renderCurrentStep();
      if (scenario.onSuccess) this.onSuccess(scenario.onSuccess);
    } else {
      this._registerWrong(scenario.onWrong || "別のマスです。もう一度探してみましょう。");
    }
  }

  /**
   * 召喚・排除（単発アクション）を検証して pendingActions に積み、「ターン終了」待ちにする。
   * 正解の単発アクションはこの時点で期待値照合し、間違いなら即フィードバックする。
   */
  _tryAddSingle(scenario, action) {
    const result = tryApplyActions(scenario, [action]);
    if (!result.ok) {
      this._registerWrong(scenario.onWrong || result.error);
      return;
    }
    if (!matchesExpectation([action], scenario.expect, scenario.boardSize)) {
      this._registerWrong(scenario.onWrong || "この手は正解ではありません。もう一度試してみましょう。");
      return;
    }

    this.pendingActions = [action];
    this._workingBoard = result.board;
    this.selected = null;
    this._renderCurrentStep();
    this._hintAfterAction();
  }

  /**
   * 移動アクションを1つ組み立て、枚数選択が必要なら promptCount を await してから
   * pendingActions に蓄積する。連続移動可能ならその場で継続し、「ターン終了」待ちにする。
   */
  async _beginOrContinueMove(scenario, from, to) {
    const isContinuing = this.pendingActions.length > 0;
    const board = this.displayBoard;
    const max = maxMovableCount(board, from, scenario.currentPlayer, isContinuing);

    if (max <= 0) {
      this._registerWrong(
        isContinuing
          ? "連続移動では、自分の駒を1つ残す必要があります。"
          : scenario.onWrong || "その駒は動かせません。"
      );
      return;
    }

    let count = 1;
    if (max >= 2) {
      const chosen = await this.promptCount(max);
      if (!chosen || chosen < 1 || chosen > max) {
        // キャンセル: 選択を解除するのみで pendingActions はそのまま維持する。
        this.selected = null;
        this._renderCurrentStep();
        return;
      }
      count = chosen;
    }

    // 実際に board 上でこの移動が合法かを simulateTurn で検証する（累積アクションで）。
    const action = buildMoveAction(from, to, count);
    const candidateActions = [...this.pendingActions, action];
    const result = tryApplyActions(scenario, candidateActions);

    if (!result.ok) {
      this._registerWrong(scenario.onWrong || result.error);
      return;
    }

    this.pendingActions = candidateActions;
    this._workingBoard = result.board;

    // 移動先に自駒が2枚以上残るなら連続移動が可能 → 続けて選択状態にする。
    const destStack = normalizeStack(result.board?.[to.r]?.[to.c]);
    let ownTop = 0;
    for (let i = destStack.length - 1; i >= 0; i--) {
      if (destStack[i] === scenario.currentPlayer) ownTop++;
      else break;
    }
    this.selected = ownTop > 1 ? { r: to.r, c: to.c } : null;
    this._renderCurrentStep();
    this._hintAfterAction();
  }

  /**
   * 「ターン終了」ボタンから呼ばれる。pendingActions を確定し、期待値照合の上で進む。
   */
  endTurn() {
    const scenario = this.currentScenario;
    if (this.pendingActions.length === 0) return;

    const result = tryApplyActions(scenario, this.pendingActions);
    if (!result.ok) {
      this._registerWrong(scenario.onWrong || result.error);
      return;
    }

    if (!matchesExpectation(this.pendingActions, scenario.expect, scenario.boardSize)) {
      this._registerWrong(scenario.onWrong || "この手は正解ではありません。もう一度試してみましょう。");
      return;
    }

    this.onKifu(result.notation);
    this._finishTurn(scenario);
  }

  _clearPending() {
    this.pendingActions = [];
    this._workingBoard = null;
    this.selected = null;
  }

  _finishTurn(scenario) {
    this._clearPending();
    if (scenario.onSuccess) {
      this.onSuccess(scenario.onSuccess);
    }
    this.advance();
  }

  advance() {
    this._clearPending();
    if (this.stepIndex < this.scenarios.length - 1) {
      this.stepIndex += 1;
      this._saveProgress();
      this.wrongCount = 0;
      this._resetStepState();
      this._renderCurrentStep();
    } else {
      this._saveProgress();
      this.onMessage({ ...this.currentScenario, message: "チュートリアル完了です！お疲れさまでした。" });
    }
  }
}
