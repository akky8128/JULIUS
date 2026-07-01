import { ref, get } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

// 未ログイン、またはニックネーム未設定のユーザーをホームへ戻す共通ガード。
// 対象ページ(ホーム・profile.html・game.htmlの観戦モードを除く)のonAuthStateChangedの先頭で呼び出す。
export async function requireAuthAndNickname(user, database) {
    if (!user) {
        alert("ログインが必要です。トップページに戻ります。");
        window.location.href = 'index.html';
        return false;
    }

    const nicknameSnap = await get(ref(database, `users/${user.uid}/profile/nickname`));
    if (!nicknameSnap.exists() || !nicknameSnap.val()) {
        alert("ニックネームの設定が必要です。トップページに戻ります。");
        window.location.href = 'index.html';
        return false;
    }

    return true;
}
