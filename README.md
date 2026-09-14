# Ukeja Layer

自作ボードゲーム「ウケジャレイヤー(Ukeja Layer)」のモノレポ。

| ディレクトリ | 内容 |
|---|---|
| [`core/`](core/) | ルール・着手生成・探索・NNUE/AlphaZero推論の**正本**(ブラウザ/Node両対応の純ESM) |
| [`julius/`](julius/) | オンライン対戦サイト **JuliUs.** — Japan Ukeja Layer Innovative Users Service (Firebase) |
| [`research/`](research/) | AI研究(NNUE・AlphaZero・解析ツール)。デプロイ対象外 |

依存の向きは `julius → core ← research` の一方通行。`julius/` と `research/` は互いを参照しない。

## よく使うコマンド(リポジトリ直下で実行)

```bash
npm test               # core / julius / research の全テスト
npm run test:core      # ルール・AIコアのテスト
npm run sync           # core/ を julius/public/js/core と julius/functions/core へコピー
npm run lab            # http://localhost:8777/research/lab/analysis.html などを配信
npm run serve:julius   # sync してから Firebase エミュレーター起動
npm run deploy:julius  # Firebase へデプロイ(predeploy で自動 sync)
```

## core/ の変更ルール

- ルール・AIを変更するときは **`core/` だけ**を編集する。
- `julius/public/js/core/` と `julius/functions/core/` は `julius/scripts/sync-core.mjs` が生成するコピー(git管理外・編集禁止)。
  Firebase Hosting / Functions がそれぞれのディレクトリ外を参照できないため複製している。
