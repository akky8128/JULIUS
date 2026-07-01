import { auth, database, functions } from './firebase.js';
import { onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { ref, get, onValue } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";
import { httpsCallable } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-functions.js";

const dismissInvitationFn = httpsCallable(functions, 'dismissInvitation');
const dismissFollowNotificationFn = httpsCallable(functions, 'dismissFollowNotification');

const notificationSound = new Audio('sounds/notification.mp3');
let invitationsListener = null;
let followNotificationsListener = null;
let latestInvitations = {};
let latestFollowNotifications = {};

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
        document.getElementById('user-menu-dropdown')?.classList.add('hidden');
    });
    document.addEventListener('click', () => {
        dropdown.classList.add('hidden');
        document.getElementById('user-menu-dropdown')?.classList.add('hidden');
    });
    dropdown.addEventListener('click', e => e.stopPropagation());

    onAuthStateChanged(auth, async (user) => {
        if (invitationsListener) invitationsListener();
        if (followNotificationsListener) followNotificationsListener();
        latestInvitations = {};
        latestFollowNotifications = {};

        if (user) {
            const userProfileRef = ref(database, `users/${user.uid}/profile`);
            const snapshot = await get(userProfileRef);
            const userNickname = snapshot.exists() ? snapshot.val().nickname : null;

            authStatusContainer.innerHTML = '';

            const menuWrapper = document.createElement('div');
            menuWrapper.className = 'relative';

            const menuButton = document.createElement('button');
            menuButton.id = 'user-menu-button';
            menuButton.className = `flex items-center font-bold focus:outline-none ${userNickname ? 'text-gray-700 hover:text-blue-600' : 'text-yellow-600 hover:text-yellow-700'}`;

            // ニックネームはユーザー入力を含むためtextContentで安全に挿入する
            const nameSpan = document.createElement('span');
            nameSpan.textContent = userNickname || 'ニックネームを設定';
            menuButton.appendChild(nameSpan);

            const caret = document.createElement('i');
            caret.className = 'fas fa-caret-down ml-1 text-xs';
            menuButton.appendChild(caret);

            const menuDropdown = document.createElement('div');
            menuDropdown.id = 'user-menu-dropdown';
            menuDropdown.className = 'hidden absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl z-20';

            const profileLink = document.createElement('a');
            profileLink.href = `profile.html?uid=${encodeURIComponent(user.uid)}`;
            profileLink.className = 'block px-4 py-3 hover:bg-gray-100 border-b text-gray-800';
            profileLink.textContent = 'プロフィールを見る';

            const logoutButton = document.createElement('button');
            logoutButton.className = 'w-full text-left px-4 py-3 hover:bg-gray-100 text-red-600';
            logoutButton.textContent = 'ログアウト';
            logoutButton.addEventListener('click', () => {
                signOut(auth).then(() => {
                    window.location.href = 'index.html';
                }).catch(console.error);
            });

            menuDropdown.appendChild(profileLink);
            menuDropdown.appendChild(logoutButton);
            menuWrapper.appendChild(menuButton);
            menuWrapper.appendChild(menuDropdown);
            authStatusContainer.appendChild(menuWrapper);

            menuButton.addEventListener('click', (e) => {
                e.stopPropagation();
                dropdown.classList.add('hidden');
                menuDropdown.classList.toggle('hidden');
            });
            menuDropdown.addEventListener('click', e => e.stopPropagation());

            const invitationsRef = ref(database, `invitations/${user.uid}`);
            invitationsListener = onValue(invitationsRef, (snapshot) => {
                latestInvitations = snapshot.val() || {};
                updateNotifications(latestInvitations, latestFollowNotifications);
            });

            const followNotificationsRef = ref(database, `followNotifications/${user.uid}`);
            followNotificationsListener = onValue(followNotificationsRef, (snapshot) => {
                latestFollowNotifications = snapshot.val() || {};
                updateNotifications(latestInvitations, latestFollowNotifications);
            });
        } else {
            authStatusContainer.innerHTML = `<a href="profile.html" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg text-sm transition">ログイン</a>`;
            updateNotifications({}, {});
        }
    });
}

async function updateNotifications(invitations, followNotifications) {
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
            dismissInvitationFn({gameId}).catch((error) => {
                console.error("Failed to dismiss stale invitation:", error);
            });
        }
    }

    sessionStorage.setItem('notifiedGameIds', JSON.stringify(notifiedGameIds));

    // --- フォロー通知 ---
    const followEntries = Object.entries(followNotifications || {})
        .sort(([, a], [, b]) => (b.timestamp || 0) - (a.timestamp || 0));
    let notifiedFollowerIds = JSON.parse(sessionStorage.getItem('notifiedFollowerIds')) || [];

    followEntries.forEach(([followerUid]) => {
        if (!notifiedFollowerIds.includes(followerUid)) {
            shouldPlaySound = true;
            notifiedFollowerIds.push(followerUid);
        }
    });

    sessionStorage.setItem('notifiedFollowerIds', JSON.stringify(notifiedFollowerIds));

    if (shouldPlaySound) {
        notificationSound.play().catch(() => {});
    }

    const totalCount = validInvitations.length + followEntries.length;
    if (totalCount > 0) {
        badge.classList.remove('hidden');
        badge.textContent = totalCount;

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
            // クリックして通知内容を確認したら、その通知は消す。
            // <a>のデフォルト遷移を先に走らせるとページ遷移によって
            // 削除リクエストが完了前にキャンセルされることがあるため、
            // 一旦遷移を止めて削除完了後に手動で遷移する。
            item.addEventListener('click', (e) => {
                e.preventDefault();
                dismissInvitationFn({ gameId: inv.gameId })
                    .catch((error) => {
                        console.error("Failed to dismiss invitation:", error);
                    })
                    .finally(() => {
                        window.location.href = item.href;
                    });
            });
            list.appendChild(item);
        });

        followEntries.forEach(([followerUid, info]) => {
            const item = document.createElement('a');
            item.href = `profile.html?uid=${encodeURIComponent(followerUid)}`;
            item.className = 'block p-3 hover:bg-gray-100 border-b';

            const nameP = document.createElement('p');
            nameP.className = 'font-semibold';
            nameP.textContent = `${info.followerNickname || 'ゲスト'}さん`;

            const descP = document.createElement('p');
            descP.className = 'text-sm text-gray-600';
            descP.textContent = 'にフォローされました。';

            item.appendChild(nameP);
            item.appendChild(descP);
            // クリックして通知内容を確認したら、その通知は消す。
            // 遷移を先に止めて、削除完了後に手動で遷移する(理由は上と同じ)。
            item.addEventListener('click', (e) => {
                e.preventDefault();
                dismissFollowNotificationFn({ followerUid })
                    .catch((error) => {
                        console.error("Failed to dismiss follow notification:", error);
                    })
                    .finally(() => {
                        window.location.href = item.href;
                    });
            });
            list.appendChild(item);
        });
    } else {
        badge.classList.add('hidden');
        list.innerHTML = '<p class="p-3 text-sm text-gray-500">新しい通知はありません。</p>';
    }
}
