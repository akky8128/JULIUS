// ===================================================================
// チュートリアル本編シナリオ集（章1〜6 + 締め）。
// board は denormalized 形式（空セル=0, 駒あり=下から上への配列）。
//
// ステップの種類:
//  - kind:'info'          … 解説のみ。盤面クリックでは進まず「次へ」で進む。
//  - expect.type:'tap'    … 座標を覚えるための「タップするだけ」の練習。正解後は「次へ」で進む。
//  - それ以外の action     … 召喚 / 排除 / 移動。「ターン終了」を押して手を確定すると進む。
//
// 補助フィールド:
//  - revealHighlightAfterWrong: N … N回間違えるまで黄色いハイライトを出さない（考えさせる）。
//  - allowEliminateMode:true      … 移動/排除の切り替えトグルをUIに出す（排除の章で使う）。
//  - defaultActionMode:'eliminate'… ステップ開始時の操作モード。
//
// プリセット盤面はすべて tools/tutorial.test.mjs で simulateTurn を通して
// 正当性（合法手であること・章6は勝利判定が立つこと）を検証している。
// ===================================================================

function emptyBoard(size) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
}

const BOARD_SIZE = 4;

// ───────────────────────── 第1章: 盤と座標 ─────────────────────────

const chapter1Info = {
  id: "ch1-info-1",
  kind: "info",
  chapter: 1,
  chapterTitle: "盤と座標",
  title: "盤と座標の読み方",
  board: emptyBoard(BOARD_SIZE),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 0, black: 0 },
  message:
    "ウケジャレイヤーの盤は基本4×4のマス目です。横の列にはa〜dのアルファベット、縦の段には1〜4の数字がついています。" +
    "例えば「c3」は、左からc列・下から3段目のマスのことです。座標は「列＋段」の順で読みます。",
  highlights: [],
};

// 座標を覚えるためのタップ練習を3問。最初はヒント（黄色枠）を出さず、2回間違えたら出す。
function tapScenario(id, coord) {
  return {
    id,
    kind: "action",
    chapter: 1,
    chapterTitle: "盤と座標",
    title: `「${coord}」はどこ？`,
    board: emptyBoard(BOARD_SIZE),
    boardSize: BOARD_SIZE,
    currentPlayer: "white",
    summonCounts: { white: 0, black: 0 },
    message: `「${coord}」のマスをタップしてみましょう。列（a〜d）と段（1〜4）を手がかりに探してください。`,
    highlights: [coord],
    revealHighlightAfterWrong: 2,
    expect: { type: "tap", at: coord },
    onWrong: "そのマスではありません。列（a〜d）と段（1〜4）をもう一度確認してみましょう。",
    onSuccess: `正解！これが「${coord}」です。「次へ」で進みましょう。`,
  };
}

const chapter1TapC3 = tapScenario("ch1-action-1", "c3");
const chapter1TapD2 = tapScenario("ch1-action-2", "d2");
const chapter1TapA4 = tapScenario("ch1-action-3", "a4");

// ───────────────────────── 第2章: 召喚 ─────────────────────────

const chapter2Info = {
  id: "ch2-info-1",
  kind: "info",
  chapter: 2,
  chapterTitle: "召喚",
  title: "駒を盤に出す「召喚」",
  board: emptyBoard(BOARD_SIZE),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 0, black: 0 },
  message:
    "対局の始まりは、空いているマスに自分の駒を1つ置く「召喚」から。召喚できる回数は白・黒それぞれ8回までです。" +
    "先手は白番。まずは白が召喚するところから対局が始まります。",
  highlights: [],
};

const chapter2Action = {
  id: "ch2-action-1",
  kind: "action",
  chapter: 2,
  chapterTitle: "召喚",
  title: "召喚してみよう",
  board: emptyBoard(BOARD_SIZE),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 0, black: 0 },
  message:
    "光っているマス「b2」に白の駒を召喚してみましょう。駒を置いたら「ターン終了」ボタンを押して手を確定します。",
  highlights: ["b2"],
  expect: { type: "summon", at: "b2" },
  onWrong: "空いているマス（光っているb2）をタップして召喚してください。",
  onSuccess: "召喚成功！これで白の駒が盤に1つ出ました。",
};

// ───────────────────────── 第3章: 移動とスタック ─────────────────────────

const chapter3Info = {
  id: "ch3-info-1",
  kind: "info",
  chapter: 3,
  chapterTitle: "移動とスタック",
  title: "駒の移動と「スタック」",
  board: emptyBoard(BOARD_SIZE),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  message:
    "駒は縦か横に1マスだけ移動できます。移動先に駒があれば、その上に重なります。" +
    "この「重ねられた駒のまとまり」を「スタック」と呼びます。自分の駒が相手の駒の上に乗ると、" +
    "そのマスは自分の駒として扱われ、相手の動きを封じることができます。",
  highlights: [],
};

function chapter3Board() {
  const board = emptyBoard(BOARD_SIZE);
  // b2 = {r:2,c:1} に白、b3 = {r:1,c:1} に黒。
  board[2][1] = ["white"];
  board[1][1] = ["black"];
  return board;
}

const chapter3Action = {
  id: "ch3-action-1",
  kind: "action",
  chapter: 3,
  chapterTitle: "移動とスタック",
  title: "相手の駒の上にスタックを作ろう",
  board: chapter3Board(),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  message:
    "b2の白い駒をタップして選択し、真上のb3にある黒い駒に重ねてみましょう。重ねたら「ターン終了」で確定します。",
  highlights: ["b2", "b3"],
  expect: { type: "move", from: "b2", to: "b3" },
  onWrong: "b2の駒を選んでb3へ移動させ、黒の駒の上に重ねてください。",
  onSuccess: "スタックができました！白が黒の上に乗ったので、このマスは白の駒として扱われます。",
};

// ───────────────────────── 第4章: 連携と連続移動 ─────────────────────────

const chapter4Info = {
  id: "ch4-info-1",
  kind: "info",
  chapter: 4,
  chapterTitle: "連携と連続移動",
  title: "「連携」と連続移動",
  board: emptyBoard(BOARD_SIZE),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  message:
    "スタックが縦か横に複数マス繋がっていて、それぞれの一番上の色が同じ状態を「連携」と呼びます。" +
    "連携があると、その列（や行）に沿って駒を1手で遠くまで運べます。これは移動を1手番の中で続けて行える" +
    "「連続移動」のおかげです。連続移動には2つの決まりがあります。①直前に移動した先のマスからしか動かせない " +
    "②必ず自分の駒を1つそのマスに残さなければならない。",
  highlights: [],
};

function chapter4LinkBoard() {
  const board = emptyBoard(BOARD_SIZE);
  // b1に白、b2・b3に「黒の上に白が乗った」スタック、b4に黒。b1〜b3は最上段が白＝連携。
  board[3][1] = ["white"]; // b1
  board[2][1] = ["black", "white"]; // b2
  board[1][1] = ["black", "white"]; // b3
  board[0][1] = ["black"]; // b4
  return board;
}

const chapter4LinkAction = {
  id: "ch4-action-1",
  kind: "action",
  chapter: 4,
  chapterTitle: "連携と連続移動",
  title: "連携で駒を遠くへ運ぼう",
  board: chapter4LinkBoard(),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  message:
    "b1・b2・b3は一番上がどれも白で「連携」しています。b1の白を、b2→b3→b4と1マスずつ連続で動かして、" +
    "b4の黒に乗せてみましょう。各マスには白を1つ残しながら進みます。最後に「ターン終了」で確定します。",
  highlights: ["b1", "b2", "b3", "b4"],
  expect: {
    type: "moveChain",
    steps: [
      { from: "b1", to: "b2" },
      { from: "b2", to: "b3" },
      { from: "b3", to: "b4" },
    ],
  },
  onWrong: "b1→b2→b3→b4の順に1マスずつ連続で動かし、「ターン終了」を押してください。各マスに白を1つ残す必要があります。",
  onSuccess: "連携のおかげで、白の駒が一気にb4まで届きました！これが連続移動の機動力です。",
};

const chapter4ConcInfo = {
  id: "ch4-info-2",
  kind: "info",
  chapter: 4,
  chapterTitle: "連携と連続移動",
  title: "「濃縮」— 強さと危うさ",
  board: emptyBoard(BOARD_SIZE),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  message:
    "自分の駒が連続して重なっている状態を「濃縮」と呼びます。濃縮の一番のメリットは、連続移動が強力になること。" +
    "重なった枚数だけ駒を配れるので、1手で何マスも進んだり、複数のマスに影響を与えたりできる高い機動力を生みます。" +
    "一方で危うさもあります。濃縮を相手に「踏まれる」（一番上を取られる）と、重なった駒の塊ごと相手に奪われ、" +
    "一気に大きなマテリアル（駒の損得）の差がついてしまいます。強力だからこそ慎重に扱いましょう。",
  highlights: [],
};

function chapter4ConcBoard() {
  const board = emptyBoard(BOARD_SIZE);
  // a3に白3枚の濃縮、d3に黒1枚。a3〜d3は同じ3段目の行。
  board[1][0] = ["white", "white", "white"]; // a3
  board[1][3] = ["black"]; // d3
  return board;
}

const chapter4ConcAction = {
  id: "ch4-action-2",
  kind: "action",
  chapter: 4,
  chapterTitle: "連携と連続移動",
  title: "濃縮の機動力を体験しよう",
  board: chapter4ConcBoard(),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  message:
    "a3には白が3枚重なった「濃縮」があります。この3枚を使い、b3→c3→d3へと1枚ずつ配りながら連続移動して、" +
    "d3の黒に乗せましょう（枚数を聞かれたら、そのまま「決定」でOKです）。最後に「ターン終了」で確定します。",
  highlights: ["a3", "b3", "c3", "d3"],
  expect: {
    type: "moveChain",
    steps: [
      { from: "a3", to: "b3" },
      { from: "b3", to: "c3" },
      { from: "c3", to: "d3" },
    ],
  },
  onWrong: "a3の3枚を使い、b3→c3→d3の順に連続で動かして、「ターン終了」を押してください。各マスに白を1つ残して進みます。",
  onSuccess: "見事です！濃縮した駒が一気に3マス先まで届きました。これが濃縮の機動力です。",
};

// ───────────────────────── 第5章: 排除 ─────────────────────────

const chapter5Info = {
  id: "ch5-info-1",
  kind: "info",
  chapter: 5,
  chapterTitle: "排除",
  title: "相手の駒を取り除く「排除」",
  board: emptyBoard(BOARD_SIZE),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  message:
    "白・黒ともに8回の召喚をすべて終えると、「排除」ができるようになります。" +
    "自分の駒が一番上にあるマスを選ぶと、その下にある相手の駒（一番上にある相手駒）を1つ取り除けます。" +
    "移動と排除は操作パネルの切り替えで選べます。",
  highlights: [],
};

function chapter5Board() {
  const board = emptyBoard(BOARD_SIZE);
  // c3 = {r:1,c:2} に黒の上に白が乗ったスタック。
  board[1][2] = ["black", "white"];
  return board;
}

const chapter5Action = {
  id: "ch5-action-1",
  kind: "action",
  chapter: 5,
  chapterTitle: "排除",
  title: "排除してみよう",
  board: chapter5Board(),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  allowEliminateMode: true,
  defaultActionMode: "eliminate",
  message:
    "操作は「排除」モードになっています。c3では白が一番上に乗っています。c3をタップして下の黒の駒を排除し、" +
    "「ターン終了」で確定しましょう。",
  highlights: ["c3"],
  expect: { type: "eliminate", at: "c3" },
  onWrong: "自分の駒が一番上にあるc3をタップして排除してください（操作モードが「排除」になっているか確認しましょう）。",
  onSuccess: "排除成功！相手の駒を1つ盤から取り除きました。",
};

// ───────────────────────── 第6章: 勝利条件 ─────────────────────────

const chapter6Info = {
  id: "ch6-info-1",
  kind: "info",
  chapter: 6,
  chapterTitle: "勝利条件",
  title: "勝利条件",
  board: emptyBoard(BOARD_SIZE),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  message:
    "ウケジャレイヤーの勝利条件はシンプルです。盤上にある相手の駒すべての上に自分の駒が乗った状態になれば、" +
    "相手は動かせる駒がなくなり、あなたの勝ちです。相手の駒を1つずつ覆い尽くしていくことを目指しましょう。",
  highlights: [],
};

function chapter6Board() {
  const board = emptyBoard(BOARD_SIZE);
  // d3 = {r:1,c:3} に白、d4 = {r:0,c:3} に黒（黒はこれが最後の1枚）。
  board[1][3] = ["white"];
  board[0][3] = ["black"];
  return board;
}

const chapter6Action = {
  id: "ch6-action-1",
  kind: "action",
  chapter: 6,
  chapterTitle: "勝利条件",
  title: "あと1手で勝利！",
  board: chapter6Board(),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 8, black: 8 },
  message:
    "黒の駒はd4に残った1つだけです。d3の白い駒をd4へ動かして黒の上に乗せれば、黒は動ける駒がなくなり、あなたの勝ちです。" +
    "移動したら「ターン終了」で確定しましょう。",
  highlights: ["d3", "d4"],
  expect: { type: "move", from: "d3", to: "d4" },
  onWrong: "d3の駒を選んでd4へ移動させ、黒の駒の上に乗ってください。",
  onSuccess: "勝利です！相手の駒すべての上にあなたの駒が乗りました。これがウケジャレイヤーのゴールです。",
  checkWin: true,
};

// ───────────────────────── 締め ─────────────────────────

const closingInfo = {
  id: "closing-info-1",
  kind: "info",
  chapter: 7,
  chapterTitle: "おわりに",
  title: "チュートリアル完了！",
  board: emptyBoard(BOARD_SIZE),
  boardSize: BOARD_SIZE,
  currentPlayer: "white",
  summonCounts: { white: 0, black: 0 },
  message:
    "お疲れさまでした！これでウケジャレイヤーの基本ルールはすべて学び終えました。" +
    "次はCPUと対戦してみましょう（近日公開）。今すぐ対局を作りたい場合は「create.html」から新しい対局を作成できます。",
  highlights: [],
};

export const scenarios = [
  chapter1Info,
  chapter1TapC3,
  chapter1TapD2,
  chapter1TapA4,
  chapter2Info,
  chapter2Action,
  chapter3Info,
  chapter3Action,
  chapter4Info,
  chapter4LinkAction,
  chapter4ConcInfo,
  chapter4ConcAction,
  chapter5Info,
  chapter5Action,
  chapter6Info,
  chapter6Action,
  closingInfo,
];
