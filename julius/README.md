# JuliUs.

Ukeja Layer のオンライン対戦サイト(Firebase Hosting + Realtime Database + Cloud Functions)。

```
julius/
├── firebase.json / .firebaserc / database.rules.json
├── public/          Hosting の公開ルート(ここに置いたものだけが公開される)
│   ├── *.html       ページ
│   ├── partials/    header.html / footer.html(js/header.js, js/footer.js が fetch)
│   ├── js/          ページ用スクリプト・tutorial/・workers/aiWorker.js(CPU対戦)
│   ├── js/core/     ← ../../core の生成コピー(git管理外・編集禁止)
│   ├── assets/      画像・アイコン・ルール説明gif
│   └── sounds/
├── functions/       Cloud Functions(index.js, idaMove.js, models/gen137.*)
│   └── core/        ← ../../core の生成コピー(git管理外・編集禁止)
├── scripts/         sync-core.mjs, migrateCurrentNode.mjs(RTDB移行)
├── test/            tutorial.test.mjs
├── design/          公開しない素材
└── backups/         RTDBエクスポート等(git管理外・公開禁止)
```

## 開発

```bash
node scripts/sync-core.mjs                  # core を public/ と functions/ へコピー(初回・core変更後に必須)
firebase emulators:start                    # julius/ で実行
npm --prefix functions test                 # pretest で sync される
python3 -m http.server 8778 -d public       # Firebase 不要な画面(トップ・チュートリアル)の簡易確認
```

`firebase deploy` 時は `firebase.json` の predeploy で `sync-core.mjs` が自動実行される。
