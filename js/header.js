import { auth, database } from './firebase.js';
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { ref, get, onValue, remove } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

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
        if (invitationsListener) invitationsListener();

        if (user) {
            const userProfileRef = ref(database, `users/${user.uid}/profile`);
            const snapshot = await get(userProfileRef);
            const userNickname = snapshot.exists() ? snapshot.val().nickname : null;

            authStatusContainer.innerHTML = '';
            const authLink = document.createElement('a');
            authLink.href = `profile.html?uid=${encodeURIComponent(user.uid)}`;
            if (userNickname) {
                authLink.className = 'font-bold text-gray-700 hover:text-blue-600';
                authLink.textContent = userNickname;
            } else {
                authLink.className = 'font-bold text-yellow-600 hover:text-yellow-700';
                authLink.textContent = 'ニックネームを設定';
            }
            authStatusContainer.appendChild(authLink);

            const invitationsRef = ref(database, `invitations/${user.uid}`);
            invitationsListener = onValue(invitationsRef, (snapshot) => {
                updateNotifications(snapshot.val() || {});
            });
        } else {
            authStatusContainer.innerHTML = `<a href="profile.html" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg text-sm transition">ログイン</a>`;
            updateNotifications({});
        }
    });
}

async function updateNotifications(invitations) {
    const badge = document.getElementById('notification-badge');
    const list = document.getElementById('notification-list');
    const bellContainer = document.getElementById('notification-container');

    if (!auth.currentUser) {
        bellContainer.classList.add('hidden');
        return;
    }
    bellContainer.classList.remove('hidden');

    list.innerHTML = '';
    const validInvitations = [];
    let notifiedGameIds = JSON.parse(sessionStorage.getItem('notifiedGameIds')) || [];
    let shouldPlaySound = false;

    for (const gameId in invitations) {
        const inv = invitations[gameId];
        const gameStatusRef = ref(database, `games/${gameId}/meta/status`);
        const gameSnap = await get(gameStatusRef);

        if (gameSnap.exists() && gameSnap.val() === 'waiting') {
            validInvitations.push({ gameId, ...inv });
            if (!notifiedGameIds.includes(gameId)) {
                shouldPlaySound = true;
                notifiedGameIds.push(gameId);
            }
        } else {
            remove(ref(database, `invitations/${auth.currentUser.uid}/${gameId}`));
        }
    }

    sessionStorage.setItem('notifiedGameIds', JSON.stringify(notifiedGameIds));

    if (shouldPlaySound) {
        notificationSound.play().catch(() => {});
    }

    if (validInvitations.length > 0) {
        badge.classList.remove('hidden');
        badge.textContent = validInvitations.length;
        validInvitations.forEach(inv => {
            const item = document.createElement('a');
            item.href = `join.html?id=${inv.gameId}`;
            item.className = 'block p-3 hover:bg-gray-100 border-b';

            const nameP = document.createElement('p');
            nameP.className = 'font-semibold';
            nameP.textContent = `${inv.inviterNickname || 'ゲスト'}さん`;

            const descP = document.createElement('p');
            descP.className = 'text-sm text-gray-600';
            descP.textContent = 'から対局に招待されました。';

            item.appendChild(nameP);
            item.appendChild(descP);
            list.appendChild(item);
        });
    } else {
        badge.classList.add('hidden');
        list.innerHTML = '<p class="p-3 text-sm text-gray-500">新しい通知はありません。</p>';
    }
}
