import { createContext, useContext, useEffect, useState } from "react";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  getAuth

} from "firebase/auth";
import { auth, db } from "../../firebase";
import {
  doc,
  getDoc,
  setDoc,
  updateDoc,
  getDocs,
  writeBatch,
  collection,
} from "firebase/firestore";
import { useNavigate } from "react-router-dom";
import { getDatabase, ref, set } from "firebase/database";

const AuthContext = createContext();

export const AuthContextProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [signinError, setSigninError] = useState();
  const [signupError, setSignupError] = useState();
  const navigate = useNavigate();


  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      setUser(user);
    });

    return unsubscribe;
  }, []);
  const createPatients = async (patientInfo) => {
    try {
      const userRef = doc(collection(db, "newUsers")); // Generate a new document with a unique ID
      await setDoc(userRef, {
        name: patientInfo.name,
        email: patientInfo.email,
        userType: patientInfo.userType,
        controllerID: patientInfo.controllerID,
        status: "pending", // Account status set to pending
        birthday: "",
        nic: "",
        mobile: "",
        address: "",
        profilepic: "",
        isUpdate: false,
        securityid: "",
        reminders:[]
      });
      console.log("Patient account waiting for confirmation");
      alert("Patient account waiting for confirmation");
    } catch (error) {
      console.log("Error creating patient entry:", error);
    }
  };
  

  const createPatient = async (patientInfo) => {
    try {
      // const userCredential = await createUserWithEmailAndPassword(
      //   auth,
      //   patientInfo.email,
      //   patientInfo.password
      // );
      // const user_ID = userCredential.user.uid;
      const userRef = doc(db, "newUsers");
      await setDoc(userRef, {
        name: patientInfo.name,
        email: patientInfo.email,
        userType: patientInfo.userType,
        controllerID:patientInfo.controllerID,
        // user_ID,
        birthday: "",
        nic: "",
        mobile: "",
        address: "",
        profilepic: "",
        isUpdate: false,
        securityid: "",
      });
      console.log("Patient account waiting for confermation");
      alert("Patient account waiting for confermation");
      // alert("User Account Create Success");
      window.location.reload();
    } catch (error) {
      setSignupError(error.message);
      console.log("error", error);
    }
  };

  const createCareGiver = async (careGiverInfo) => {
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        careGiverInfo.email,
        careGiverInfo.password
      );
      const user_ID = userCredential.user.uid;
      const userRef = doc(db, "users", user_ID);
      await setDoc(userRef, {
        name: careGiverInfo.name,
        email: careGiverInfo.email,
        userType: careGiverInfo.userType,
        controllerID: careGiverInfo.controllerID,
        password: careGiverInfo.password,
        user_ID,
        registernumber: "",
        nic: "",
        mobile: "",
        address: "",
        profilepic: "",
      });
      console.log("Care Giver created");
      alert("User Account Create Success");
      window.location.reload();
    } catch (error) {
      setSignupError(error.message);
      console.log("error", error);
    }
  };
  const signIn = async (data) => {
    try {
        console.log('Email:', data.email);
        console.log('Password:', data.password);

        const signin = await signInWithEmailAndPassword(auth, data.email, data.password);
        console.log(signin); 

        const user_ID = auth.currentUser.uid;
        console.log(`User ID ${user_ID}`)
        const userRef = doc(db, "users", user_ID);
        const docSnap = await getDoc(userRef);
        console.log('Document Snapshot:', docSnap);

        if (docSnap.exists()) {
          const userdata = docSnap.data();
            const user_role = userdata.userType;

            if (user_role === "careGiver") {
                console.log("careGiver");
                navigate("/patientgrid");
            } else if (user_role === "patient") {
                console.log("patient");
                navigate("/");
            }
        } else {
            console.log("No such document!");
        }
    } catch (error) {
        setSigninError(error.message);
        console.log("Error during sign-in:", error.code, error.message);
    }
};

  const logOut = () => {
    signOut(auth);
    navigate("/")
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        createPatients,
        createPatient,
        createCareGiver,
        signIn,
        signinError,
        signupError,
        logOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
export const useAuth = () => {
  return useContext(AuthContext);
};
export default AuthContextProvider;
