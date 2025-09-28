import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { getDatabase, ref, get } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

const firebaseConfig = {
    apiKey: "AIzaSyBSjq7McZVw23FZisxX7wpMFrpEdX7wjBo",
    authDomain: "julius-online-a5984.firebaseapp.com",
    databaseURL: "https://julius-online-a5984-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "julius-online-a5984",
    storageBucket: "julius-online-a5984.appspot.com",
    messagingSenderId: "688172535832",
    appId: "1:688172535832:web:ba8cdf48d178b4cdcf336d"
};

// Firebaseアプリの初期化（重複しないように）
let app;
if (!window.firebaseApp) {
    app = initializeApp(firebaseConfig);
    window.firebaseApp = app;
} else {
    app = window.firebaseApp;
}

const auth = getAuth(app);
const database = getDatabase(app);

/**
 * ヘッダーを読み込み、認証状態に基づいて表示を更新する関数
 * @param {string} placeholderId - ヘッダーを挿入する要素のID
 */
export async function loadHeader(placeholderId = 'header-placeholder') {
    const headerPlaceholder = document.getElementById(placeholderId);
    if (!headerPlaceholder) return;

    // 1. header.htmlを読み込む
    const response = await fetch('header.html');
    headerPlaceholder.innerHTML = await response.text();
    
    const authStatusContainer = document.getElementById('header-auth-status');
    if (!authStatusContainer) return;

    // 2. 認証状態を監視する
    onAuthStateChanged(auth, async (user) => {
        // 3. 認証状態が確定したら、表示を更新する
        if (user) {
            // ログイン済みの場合
            const userProfileRef = ref(database, `users/${user.uid}/profile`);
            const snapshot = await get(userProfileRef);
            const userNickname = snapshot.exists() ? snapshot.val().nickname : null;

            if (userNickname) {
                authStatusContainer.innerHTML = `
                    <a href="profile.html?uid=${user.uid}" class="font-bold text-gray-700 hover:text-blue-600">${userNickname}</a>
                `;
            } else {
                authStatusContainer.innerHTML = `
                    <a href="profile.html?uid=${user.uid}" class="font-bold text-yellow-600 hover:text-yellow-700">ニックネームを設定</a>
                `;
            }
        } else {
            // 未ログインの場合
            authStatusContainer.innerHTML = `
                <a href="profile.html" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg text-sm transition">
                    ログイン
                </a>
            `;
        }
    });
}