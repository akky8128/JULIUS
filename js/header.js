import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { getDatabase, ref, get, onValue, remove } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

const firebaseConfig = {
    apiKey: "AIzaSyBSjq7McZVw23FZisxX7wpMFrpEdX7wjBo",
    authDomain: "julius-online-a5984.firebaseapp.com",
    databaseURL: "https://julius-online-a5984-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "julius-online-a5984",
    storageBucket: "julius-online-a5984.appspot.com",
    messagingSenderId: "688172535832",
    appId: "1:688172535832:web:ba8cdf48d178b4cdcf336d"
};

let app;
if (!window.firebaseApp) { app = initializeApp(firebaseConfig); window.firebaseApp = app; } 
else { app = window.firebaseApp; }

const auth = getAuth(app);
const database = getDatabase(app);
const notificationSound = new Audio('sounds/notification.mp3');

let invitationsListener = null;

export async function loadHeader(placeholderId = 'header-placeholder') {
    const headerPlaceholder = document.getElementById(placeholderId);
    if (!headerPlaceholder) return;
    const response = await fetch('header.html');
    headerPlaceholder.innerHTML = await response.text();
    
    const authStatusContainer = document.getElementById('header-auth-status');
    const bell = document.getElementById('notification-bell');
    const dropdown = document.getElementById('notification-dropdown');
    
    bell.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdown.classList.toggle('hidden');
    });
    document.addEventListener('click', () => dropdown.classList.add('hidden'));
    dropdown.addEventListener('click', e => e.stopPropagation());

    onAuthStateChanged(auth, async (user) => {
        // --- ▼▼▼ ここから修正 ▼▼▼ ---
        if (invitationsListener) invitationsListener(); // 古いリスナーを解除

        if (user) {
            // --- ログイン状態の表示を復活 ---
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

            // --- 通知機能 ---
            const invitationsRef = ref(database, `invitations/${user.uid}`);
            invitationsListener = onValue(invitationsRef, (snapshot) => {
                const invitations = snapshot.val();
                updateNotifications(invitations || {});
            });

        } else {
            // --- 未ログイン状態の表示 ---
            authStatusContainer.innerHTML = `
                <a href="profile.html" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg text-sm transition">
                    ログイン
                </a>
            `;
            // 通知もクリア
            updateNotifications({});
        }
        // --- ▲▲▲ ここまで修正 ▲▲▲ ---
    });
}

async function updateNotifications(invitations) {
    const badge = document.getElementById('notification-badge');
    const list = document.getElementById('notification-list');
    const bellContainer = document.getElementById('notification-container');

    if (!auth.currentUser) {
        bellContainer.classList.add('hidden'); // 未ログインならベルごと隠す
        return;
    }
    bellContainer.classList.remove('hidden'); // ログイン済みなら表示

    list.innerHTML = '';
    const validInvitations = [];
    let shouldPlaySound = false;
    const existingNotifications = parseInt(badge.textContent) || 0;

    for (const gameId in invitations) {
        const inv = invitations[gameId];
        const gameStatusRef = ref(database, `games/${gameId}/meta/status`);
        const gameSnap = await get(gameStatusRef);

        if (gameSnap.exists() && gameSnap.val() === 'waiting') {
            validInvitations.push({ gameId, ...inv });
        } else {
            remove(ref(database, `invitations/${auth.currentUser.uid}/${gameId}`));
        }
    }

    if (validInvitations.length > existingNotifications) {
        shouldPlaySound = true;
    }

    if (validInvitations.length > 0) {
        badge.classList.remove('hidden');
        badge.textContent = validInvitations.length;
        if (shouldPlaySound) {
            notificationSound.play().catch(() => {});
        }

        validInvitations.forEach(inv => {
            const item = document.createElement('a');
            item.href = `join.html?id=${inv.gameId}`;
            item.className = 'block p-3 hover:bg-gray-100 border-b';
            item.innerHTML = `
                <p class="font-semibold">${inv.inviterNickname || 'ゲスト'}さん</p>
                <p class="text-sm text-gray-600">があなたに対局を申し込みました。</p>
            `;
            list.appendChild(item);
        });
    } else {
        badge.classList.add('hidden');
        list.innerHTML = '<p class="p-3 text-sm text-gray-500">新しい通知はありません。</p>';
    }
}