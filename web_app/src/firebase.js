// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";
import {getStorage} from "firebase/storage"
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAbC0oQldjtj2nLQZXLFoPmerDHshsB2Gs",
  authDomain: "smart-wheel-chair-4ed28.firebaseapp.com",
  projectId: "smart-wheel-chair-4ed28",
  storageBucket: "smart-wheel-chair-4ed28.firebasestorage.app",
  messagingSenderId: "988744653855",
  appId: "1:988744653855:web:0644c6113bb7d6b584892a",
  measurementId: "G-RFELN8DS3T"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const db = getFirestore(app);
const auth = getAuth(app);
const storage = getStorage(app);
export {db, auth, storage};