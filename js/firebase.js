import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAuth, connectAuthEmulator } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { getDatabase, connectDatabaseEmulator } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";
import { getFunctions, connectFunctionsEmulator } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-functions.js";

const firebaseConfig = {
    apiKey: "AIzaSyBSjq7McZVw23FZisxX7wpMFrpEdX7wjBo",
    authDomain: "julius-online-a5984.firebaseapp.com",
    databaseURL: "https://julius-online-a5984-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "julius-online-a5984",
    storageBucket: "julius-online-a5984.appspot.com",
    messagingSenderId: "688172535832",
    appId: "1:688172535832:web:ba8cdf48d178b4cdcf336d"
};

export const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const database = getDatabase(app);
export const functions = getFunctions(app, "asia-southeast1");

const IS_LOCAL = location.hostname === "localhost" || location.hostname === "127.0.0.1";
if (IS_LOCAL) {
    connectAuthEmulator(auth, "http://127.0.0.1:9099", { disableWarnings: true });
    connectDatabaseEmulator(database, "127.0.0.1", 9000);
    connectFunctionsEmulator(functions, "127.0.0.1", 5001);
}
