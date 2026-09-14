# JuliUs NNUE 開発 引き継ぎ資料

> 自作ボードゲーム「ウケジャレイヤー(Ukeja Layer)」オンライン版「JuliUs.」の対戦CPU（NNUE評価関数）開発。
> このファイルだけで次セッションが立ち上がれるように、**この会話でしか分からない情報**を重点的にまとめた。
> 詳細な世代別勝率表・技術知見は `research/nnue/NNUE_PROGRESS.md`、要約は `~/.claude/projects/-Users-akky-Documents-coding-ChessCoach/memory/julius-nnue-status.md`。

作業ディレクトリ: リポジトリ直下(旧 `~/Documents/HTML/JULIUS` → `~/Documents/UkejaLayer`)　/　git ブランチ: `feat/nnue-cpu`

---

## 1. 現状（最重要）

- **現行最強 = `gen137`**（96/24/24 ＋**共有スタックエンコーダ E=16**、13スカラー、多様性データ`gen015_13`学習）。
  - vs heuristic プール **764/799 = 95.6%**（複数seedset合算）。前チャンピオン`gen092`（557/600=92.8%）に対し**z=2.25で統計的優位を確認して昇格**（2026-07-11未明のセッション）。
- research/lab/nnue-lab.html 既定・analysis.html 解析モデルとも **gen137** に設定済み（gen092は選択肢に残置）。
- **stack-encoder に `--encoder-max-pool` オプションを追加実装済み**（train.py/network.js、パリティ確認済み・全テスト green）。ただし9seed試して有効性は確認できず（詳細は§7.5）。

### 到達までの4段突破（頭打ちのたびにレバーを替えた）
```
gen043 (9スカラー・データループ)     vs heuristic ~91-93%  → データ追加では頭打ち
  ↓ 特徴量第2弾（終局近接スカラー4個, SCALAR_DIM 9→13）
gen071 (13スカラー)                  vs heuristic 92.9%    → データループ再び頭打ち
  ↓ アーキ変更（マス共有スタックエンコーダ）
gen092 (13スカラー＋encoder)         vs heuristic 95.0%    → 29seed+のseed探索で頭打ち
  ↓ 生成器多様化データ(gen015_13)でのseed探索
gen137 (gen092と同アーキ・gen015_13データ)  vs heuristic 95.6%  ← 現行最強
```

---

## 2. 会話でしか分からない「判定ルール」と落とし穴（最重要・繰り返し効いた）

1. **head-to-head の圧勝 ≠ 真の強さ（じゃんけん構造）**。採用可否は必ず**共通相手 heuristic への勝率**で判定する。
   - 実例多数: gen044は gen415に **100-0** だが vs heuristic は劣った。gen062/066も gen071に100%だが vs heuristicで超えず。
   - 今セッション最後の gen094 は gen092に head-to-head **81%** だが vs heuristic は **95%で完全同値**＝真の優位なし→不採用（正しく維持）。
   - head-to-head は「当たりseed探索のフィルタ」に使い、**採用は vs heuristic**。
2. **30局スクリーニングは偽陽性が多い**。採用判断は必ず**100局**規模。
   - gen084 は 30局57%→100局46%、gen1415は30局74%→74局47%、等。今セッションだけで gen044/061/062/066/084 を100局で棄却。
3. **seed分散が極端**（同一データ・同一アーキで vs 相手 0%〜100%）。**多seed学習＋選抜が必須**。当たりは1〜2本/8本程度。
4. **val loss（教師ラベルへの回帰誤差）と棋力は別物**。gen092は val loss 0.0397（gen071の0.0345より悪い）**なのに実戦は強い**。val lossで良し悪しを判断しないこと。棋力は必ず対戦で。
5. **vs heuristic は局面seedに敏感**（同一モデルでも±数pt）。世代間比較は**同一 pmatch seedset**で並べる（本セッションは base **2900** を基準に使用）。
6. **診断で切り分ける**: 頭打ちが「探索深さ不足」か「評価の質」か。gen415で時間予算スイープ(120ms=300ms=同勝率)→**評価の質が壁**と判明。よって accumulator/量子化などの eval高速化は主レバーではない。
7. **容量増(128/32/32)は無効**（9スカラー時に4seed全滅、過去も同様）。96/24/24で十分。

---

## 3. 共有スタックエンコーダ（gen092の中身・実装詳細）

**動機**: 従来 L1 `[96,256]` は「スタックの意味（誰が上・埋め・連続）」を**16マス分バラバラに学習**していて非効率。マス共有の重み（CNN的）にすると1更新あたり16倍の実効サンプルでスタック意味論を学べる＝データ効率の壁を越えた。

**構造（(A)ハイブリッド）**: 既存の全結合経路は残したまま、per-cellエンコーダを追加。
- 各マスの16次元（深さ8×色2）に**全マス共通**の `Linear(16→E=16)+clippedReLU(0..1)` を適用 → 16マス分を **sum-pool** して `p(E=16)`。
- `h1 = concat(a1[96], p[16], scalars[13])` → L2入力 109→**125**。以降 a2,a3,out は不変。
- indices→per-cell 復元: `cell = floor(idx/16)`, `slot = idx%16`, `perCell[cell][slot] += 1`（加算）。

**実装ファイル（Sonnet 5 が実装、コミット済み）**:
- `core/nnue/network.js` — forward にエンコーダ経路。`meta.stackEncoder`＋`Wenc/benc`テンソル有無で分岐。
- `research/nnue/train.py` — フラグ `--stack-encoder-dim E`（0=無効=従来）。`nn.Linear(16,E)`をper-cell適用しsum-pool。JSON/binに Wenc(tensors[8]),benc(tensors[9])を追記、`meta.stackEncoder={dim,cellSlotDim:16}`。
- `research/nnue/parity.test.mjs` — エンコーダ機のパリティ検証を追加。
- **後方互換**: `meta.stackEncoder`不在の旧モデル（gen071等）はbit一致で従来動作。JS↔PyTorchパリティ誤差 **2.1e-7**。全35テスト通過。
- ブラウザ(.bin, `createNetworkFromBuffer`)も `computeTensorLayout(meta.tensors)` が汎用なのでエンコーダ機を正しくロード（実機検証済み）。

---

## 4. 評価・学習コマンド（このセッションで整備）

**10コア並列対戦ランナー（必ずこれを使う。単一プロセスは1コア=遅い）**
```bash
# 100局を5シャード×20局に分割し並列 → Wilson CIで集計。base seedを変えると別seedset。
zsh research/nnue/bench/pmatch.sh "<white>" "<black>" <totalGames> 5 <outPrefix> <baseSeed>
# 例: gen092 vs heuristic@2900
zsh research/nnue/bench/pmatch.sh "nnue:research/models/gen092.json:d64:120" "search:d20:120" 100 5 /tmp/x 2900
cat /tmp/x.agg.txt   # 結果
# 集計だけ再実行: node research/nnue/bench/agg.mjs "<agentSubstr>" <shardFiles...>
```
- プレイヤー型: `nnue:research/models/genXXX.json:d64:120`（d=深さ上限, 120=時間ms）、`search:d20:120`（=heuristic）。
- **時間予算120ms/手・swap** が本プロジェクトの標準評価条件。

**1seedの学習＋自動スクリーニング（イベント駆動で回す）**
```bash
# エンコーダ版（batch1024で高速化）: seed, gen番号, データ, 対戦相手モデル, encDim
zsh research/nnue/bench/train_one_enc.sh <seed> <gen> research/nnue/data/gen014_13.jsonl <championModel> 16
# 非エンコーダ版
zsh research/nnue/bench/train_one.sh <seed> <gen> <data> <championModel>
```
これらを `run_in_background: true` の追跡タスクとして起動 → **完了通知でwake**して結果を見て次seedを投入、が効率的な運用（タイマーwakeより無駄がない）。

**学習の生コマンド**（train.py）
```bash
python3 research/nnue/train.py --data <jsonl> --generation <N> \
  --hidden1 96 --hidden2 24 --hidden3 24 --stack-encoder-dim 16 --seed <S> --device auto
# --generation 92 → research/models/gen092.{json,bin,golden.json}（0埋めなので gen9→gen009 に注意）
# --encoder-max-pool を付けるとsum-pool+max-poolのconcat版（2026-07-11実装、9seed試して有効性未確認）
```

---

## 5. 実行環境・性能の注意（会話でしか分からない）

- **MPS（Apple GPU）利用可・10コアCPU**。学習はGPU、データ生成/対戦はCPU。**GPU学習とCPU対戦は並列可**（同時に回せる）。
- **GPUは実質1本**: 学習を2本同時に投げるとスラッシング。学習は逐次、対戦(CPU)は並列、が基本。
- **エンコーダ学習はbatch=256だと遅い（~50分/seed）**: MPSの小カーネル起動オーバーヘッドが支配的（FLOPsは小さいのに遅い）。**batch=1024で~1/4のバッチ数に減らし高速化**（ただし bestEpoch=1 と早期収束する副作用。棋力は対戦で確認するので実害なし）。LRスケール(~4e-3)は未検証。
- **genData / selfplay は完了時に一括出力**（途中経過が出ない）。進捗は出力ファイルの有無/プロセス数で監視。
- **selfplay/train を `| tail` にパイプすると stdout がバッファされ進捗が見えない**（一度これでハマった）。ログは `> file 2>&1` で保存すること（train_one_enc.sh は修正済み）。
- **zsh**: 未クオート変数は語分割されない。ループ内の複数引数展開は明示的に。

---

## 6. データファイル（`research/nnue/data/`, 大容量・gitignoreされていないので commit 注意）

- `gen013_13.jsonl`（293万・13スカラー）= gen071の学習データ。
- `gen014_13.jsonl`（**343万・13スカラー**）= gen013_13 ＋ gen071自己対戦。**gen092の学習データ**。エンコーダ多seedはこれを使用。
- `migrate_scalars.mjs` = 既存データの indices から盤面を厳密再構成し extractFeatures を再実行してスカラーを張り替えるツール（スカラー次元を増やしたら全データをこれで移行し、既存局面で公平比較する）。
- **注意: data/ は .gitignore されていない**。commit時は**ソース・ツール・チャンピオンモデルのみ明示 add**し、巨大jsonlやsp*/shardは絶対に足さない。

---

## 7. 次の一手候補（優先度順・未着手）

**2026-07-10午後セッション（11:25〜16:00）の結果: gen092は依然チャンピオン（不変）。以下は全て試行済みで空振り**:

1. ~~E容量拡大(E24/32)~~: 不採用（screen 0%/46.7%）。
2. ~~スカラーをL1経路にも注入~~（`--scalar-in-l1`実装済み・コミット済み）: champion同seedで**screen 3.3%と壊滅的、有害と判明**。使用中止。
3. ~~純エンコーダ版(FC経路削除)~~（`--no-fc-path`実装済み・コミット済み）: E16/E24とも不採用（17.9%/10.0%）。**FC経路は有用、削除は逆効果と判明**。
4. ~~lr=1e-3/3e-4でのE16 seed探索(gen014_13データ)~~: **計29seed**試行、生存確証まで進めた4本(gen098/110/119/124)全て**統計的にgen092(94.0%[89.8,96.5]、200局プールが真の基準値)と同値**。z検定で最良のgen124(400局)でもz=0.493。**同アーキ・同データでの単純seed探索は完全に限界**。
5. **gen092自己対戦＋gen124/gen110/gen119(近傍同値モデル)を生成器に多様性データ生成**: ε0.25/orp12/augment8で32.3万レコード生成、gen014_13(343万)に追記し**`research/nnue/data/gen015_13.jsonl`(375万レコード)**を構築（コミット済み・ただし.gitignore対象外なので巨大jsonl自体はcommitしていない、再現手順は下記）。このデータで6seed学習: **gen137(seed6)=300局95.3%[92.3,97.2]・gen139(seed8)=100局96.0%[90.1,98.4]**と、旧データより一貫して高い点推定を示したが、**個別にも合算(z=0.754)でも統計的有意水準には届かず不採用**。時間切れで打ち切り。

**2026-07-11未明〜早朝セッション（自律7時間運用）の結果まとめ**:

### A. gen137昇格（成功）
gen137とgen139をそれぞれ追加確証（gen137: 799局、gen139は途中で139自体は不採用扱いに再評価）。**gen137 vs gen092の直接比較（同一seedset、vs heuristicプール）でz=2.25を達成し、gen137を新チャンピオンに昇格**。research/lab/nnue-lab.html/analysis.htmlのデフォルトも更新済み。

### B. 深ラベルデータ追加（gen016_13/gen017_13）: 不採用、ただし強いシグナルあり
- **gen016_13** = gen015_13(375万) + depth4/300ms自己対戦4shard(gen092:137:139=生成器分散、16.2万レコード追記、合計391万)。
- **gen017_13** = gen016_13 + depth5/500ms shard×2（5万レコード追記、合計396万）。
- gen016_13で**29seed**（gen142〜gen194のうちgen016系）を試行。screen(30局vs gen137)で**5候補が70%超**（gen151=70%, gen155=82.8%→不採用, gen156=70.4%, gen170=83.3%, gen178=81.5%, gen179=72.2%）と、gen015_13単体(gen137発見時)より明らかに高いヒット率。
- しかし**vs heuristicでの大規模確証（各700〜1700局）では、全候補が例外なくz<1.6に収束**:
  - gen151: 960/999=96.1%, z=0.51
  - gen156: 289/300=96.3%, z=0.52
  - gen170: 672/698=96.3%, z=0.64
  - gen178: 289/299=96.7%, z=0.77
  - **gen179: 累計1636/1698=96.35%、z=0.88〜1.49の間で終始ノイズのように振動し1.6を一度も安定して超えず**（最も粘った候補。400局を6バッチ積んでも決着せず）。
- **gen017_13（depth5追加）は8seed全滅**（最高53.3%、多くが10%未満）。depth5/500msの品質向上はむしろ逆効果の可能性。
- **合算しても不採用**: gen151+156+170+178の4候補合算で2210/2296=96.25% vs gen137 764/799=95.62%、z=0.79。深ラベルデータによる改善は「screenでは頻繁に当たるが、vs heuristicでは1pt前後の小さな上振れに過ぎず、現実的なサンプル数では確定できない」という結論。

### C. アーキ改良（sum-pool + max-pool併用）: 実装完了・不採用
- `train.py`に`--encoder-max-pool`フラグを追加実装（per-cellエンコーダのpoolingをsum-poolのみからsum+max concatに拡張、E→2E次元）。`network.js`側も対応する forward 実装済み。
- **パリティ確認済み**（誤差7e-8、全既存テストgreen、後方互換問題なし）。
- gen015_13で7seed・gen016_13で2seed（計9seed）試行、**最高57.1%で明確な当たりなし**。sum-poolのみの既存アーキを上回る証拠は得られず。

**判定ルール運用の教訓（今回で確定）**: z検定は**大きくプールしてもノイズで0.5〜1.5の間を往復することがある**（gen179で1698局積んでも決着せず）。**z≥1.6を安定して2回連続で超えない限り採用しない**のが安全。screenでの当たり率が高くても（今回29seed中5候補が70%超）、vs heuristicでの真の優位性とは別問題（じゃんけん構造の教訓が再確認された）。

**次セッションで検討すべきこと（優先度順）**:
1. ~~gen016_13データはそのまま活かしつつ、seed探索の絶対数を増やす~~ → **2026-07-11夜〜12未明のStage 0実験で「フル再ラベル」を検証したので下記§7.6参照。単純なseed数増しより優先度が下がった**。
2. **2層エンコーダ（Linear 16→32→16、深さ方向の拡張）は未着手**。max-poolでの「pooling演算子の変更」が効かなかったので、次は「表現力の拡張」を試す価値がある。
3. **lambda（教師信号のscore/result混合比、既定0.6）やscore-scaleのチューニングは未着手**。§7.6の知見（scoreとresultの内部整合性が壊れると有害）を踏まえると、**lambdaを下げてresult比重を上げる**方向が次の有力候補。
4. **完全に新しい特徴量の追加**（現在13スカラー）はgen071→gen092のブレイクスルー以来手つかず。次の特徴量候補の洗い出しが必要。
5. **AlphaZero的手法への移行を検討中**（ユーザ提案、2026-07-11夜のセッションで議論）。ターン内マイクロアクション分解（移動64+召喚/排除16+ターン終了=81ノード）によるポリシーネット設計は妥当と判断。詳細は本セッションのチャット履歴（このファイルには要約のみ）。gen137からの蒸留でコールドスタート可能。実現可能性は高い（4×4盤のため計算資源は現実的）。次の一手候補というより**並行して検討すべき別ロードマップ**。

---

## 7.6. Stage 0「フル再ラベル」実験（2026-07-11夜〜12未明・8時間セッション）: 明確に不採用、重要な負の知見

**動機**: 「gen137が頭打ちなのは浅い探索(depth3/150ms)ラベルを学習し切ったから」という仮説の統制実験。前回(gen016/017_13)は深ラベルを全体の4%だけ混ぜて希釈し不発だったため、**今回は同一局面集合(gen015_13, 469,783局面=375万レコード)を丸ごとgen137@depth5/500msで再ラベル**した。

**手法**（新規ツール `research/nnue/relabel.mjs`）:
- 各レコードの`indices`から盤面を厳密再構成（migrate_scalars.mjsと同ロジック、正当性を8000レコードで検証しmismatch=0を確認済み）。
- D4対称の8枚(sym0〜7)は同一pid・スコア共通なので**pidごとに探索1回だけ実行し使い回す**（8倍高速化、正しさも確認済み）。
- `findBestTurn`をgen137を葉評価器としてdepth5/timeBudget500msで実行し、scoreのみ差し替え。**result/indices/scalars/sym/povは一切不変**。
- 11並列ワーカーで実行、**実測スループット1.6〜1.9 pos/s/core**（当初ベンチマークの2.4 pos/s/coreより低く、想定より時間超過）。469,783局面(重複除く)の全件処理に**約8.1時間**（10コア規模でも短縮しきれなかった＝事前ベンチマークは短時間サンプルで楽観的すぎた。次回はもっと長い実測ウィンドウで見積もること）。
- 出力: `research/nnue/data/gen018_13.jsonl`（375万264レコード、gen015_13と同一局面・同一result、scoreのみ更新）。

**再ラベルは意味のある変化を生んでいた**（ノイズではない）: 元スコアとの比較で**局面の18.1%で評価の符号（有利/不利）が反転**、平均絶対差61点（score-scale=100規模で無視できない差）。

**結果: 10seed学習・screen(30局vs gen137)で全滅**:
```
gen196=50.0%, gen195=10.0%, gen197=10.0%, gen198=51.9%, gen199=10.0%,
gen201=40.0%, gen200=53.3%, gen202=3.3%, gen203=13.3%, gen204=0.0%
→ 平均24.2%、最高53.3%（gen016_13は29seed中5候補が70%超だったのと対照的）
```
**gen016_13(部分混合4%)より明確に悪く、gen137自身の学習データより悪化**。フル再ラベルは有害と結論し、gen018_13は不採用・以降の探索を打ち切った。

**なぜ逆効果だったかの考察（次セッションへの重要な仮説）**:
教師信号は `y = lambda*sigmoid(score/scale) + (1-lambda)*result`（lambda既定0.6）で、scoreとresultは本来「同じ対局・同じ着手方針の下で得られたペア」という前提で整合していた。今回はresult/着手選択（＝実際に指された手、ひいては対局の勝敗）は**旧評価器(gen092系mix, depth3/150ms)の選択のまま**残し、**score だけをgen137@depth5に差し替えた**。これにより「gen137の深い目では良さそうな局面なのに、実際の対局(旧評価器が選んだ手順)ではその後負けている」という**scoreとresultの内部矛盾**が18.1%規模で発生した可能性が高い。教師信号のλブレンドがこの矛盾したペアを学習し、ノイズの多いターゲットになったと推測される。**score付け替えとresult付け替えは常にセットで行うべきで、片方だけの差し替えは危険**という教訓。

**次に試すべきこと（優先順）**:
1. **scoreだけでなくresultも整合させる**: 再ラベルではなく、gen137@深探索で**ゲームそのものを新規に自己対戦させ直す**（genData.mjs方式、旧来路線）。時間はかかるが内部矛盾が生じない。
2. **lambdaを1.0に寄せて score のみ信頼する**（resultとの矛盾を無視する設計に変える）。あるいは逆にlambdaを0に寄せてresultのみ使う対照実験。
3. 再ラベル自体を諦めず、**「score変化が小さい(=旧評価器と深い評価が一致する)局面のみ」を対象にフィルタして再ラベル**すれば矛盾は起きにくいはず（差分が大きい18.1%を除外する等）。
4. **スループット再実測**: 1.6〜1.9 pos/s/coreは低い。findBestTurnのボトルネック調査（accumulator差分更新など）で高速化できれば同種の実験をより安く回せる。

**成果物（保持）**: `research/nnue/relabel.mjs`（再ラベルツール、他データセットにも再利用可）、`research/nnue/data/gen018_13.jsonl`（不採用データだが記録として保持）。`research/nnue/data/relabel/out_*.jsonl`と`log_*.txt`は中間生成物（`in_*`分割ファイルは容量節約のため削除済み）。

---

## 7.5. gen015_13再現手順・並列学習運用（今回セッションで確立）

**GPU(mps)+CPU(cpu)の真の並列学習**: `--device cpu`指定でtrain.pyがCPU学習でき、GPU(mps)と遜色ない速度で完走する（実測、CPU学習が遅いという心配は杞憂だった）。1本あたり5〜15分。`research/nnue/bench/run_one_dev.sh <gen> <seed> <E> <lr> <epochs> <data> <champModel> <device>`で学習+30局screenを一括実行。GPU用とCPU用を同時に2本走らせるのが基本パターン。

**多様性データ生成コマンド**（gen015_13の再現用）:
```bash
# 生成器を複数モデルに分散（単一ネット自己対戦の偏り軽減、ユーザ提案で採用）
node research/nnue/genData.mjs --games 100 --size 4 --depth 3 --timeMs 150 --maxPlies 150 \
  --seed <shardごとに変える> --openingRandomPlies 12 --epsilon 0.25 --augment 8 \
  --net research/models/gen092.json --out <shard>.jsonl   # 主力(500局・5並列)
# 同様に --net research/models/gen124.json / gen110.json / gen119.json でも各60〜100局(補助)
# pidオフセットを付けてマージ: research/nnue/bench/merge_selfplay092.py 参照(base_offset=70000000)
```
**genData.mjsは`--help`がなく、引数なしだと既定値(games:200)で単一プロセス実行してしまう**ので注意（今回誤起動して2分半CPUを無駄にした）。必ず`--games`等を明示指定すること。

## 8. 遊び方 / 動作確認

- `research/lab/nnue-lab.html` をローカルサーバ経由で開き、CPUレベルで「NNUE」→モデル選択（既定 gen092）。
- `analysis.html`（非公開・解析専用）: gen092固定で時間無制限探索・局面解析。
- モデルパスはルート絶対 `/research/models/*.json`。ブラウザは Worker で `.json`＋`.bin` をロード。
