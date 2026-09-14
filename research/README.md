# research — Ukeja Layer AI研究

JuliUs. 本体とは独立した AI 研究用ディレクトリ。デプロイされない。ルール・探索・推論は `../core/` を利用する。

```
research/
├── lab/        ブラウザで使う実験・解析ページ(analysis.html, nnue-lab.html)と Worker
├── classic/    手調整評価の自己対戦・重みチューニング(selfplay.mjs は NNUE/AZ の対戦計測にも使用)
├── nnue/       NNUE: データ生成・学習(train.py)・ベンチ・引き継ぎ資料(NNUE_HANDOFF.md, NNUE_PROGRESS.md)
├── alphazero/  AlphaZero: 自己対戦ループ・データ生成・評価ゲート・CLI対戦(PLAN.md)
└── models/     学習済みモデル(チャンピオンのみ git 管理。他は .gitignore)
```

## 約束事

- **スクリプトはリポジトリ直下をカレントディレクトリにして実行する**(パスは `research/models/...` のようにリポジトリ直下基準)。
  ```bash
  node research/classic/selfplay.mjs --games 20 --white nnue:research/models/gen137.json:d4:120 --black search
  node research/alphazero/playCli.mjs --white az:research/models/genAZ8_r83.json:120 --black search:research/models/gen137.json:120
  python3 research/nnue/train.py --data research/nnue/data/gen000.jsonl --generation 1
  ```
- 学習データ `nnue/data/`・`alphazero/data/` は数十〜数百GBあり git 管理外。
- 解析ページはリポジトリ直下を配信して開く: `npm run lab` → http://localhost:8777/research/lab/analysis.html
- `analysis.html` を JuliUs. で公開する際は `julius/public/` へ移し、import を `js/core/...`、モデルを `julius/public/models/` に置き換える。

## テスト

```bash
node --test "research/**/*.test.mjs"
(cd research/nnue && python3 -m pytest -q)
```
