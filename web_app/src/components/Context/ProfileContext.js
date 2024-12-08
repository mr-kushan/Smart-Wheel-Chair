import { createContext, useContext, useEffect, useState } from "react";
import { useAuth } from "./AuthContext";
import {
  doc,
  getDoc,
  setDoc,
  updateDoc,
  collection,
  query,
  where,
  getDocs,
} from "firebase/firestore";
import { db } from "../../firebase";

const ProfileContext = createContext();

export const ProfileContextProvider = ({ children }) => {
  const { user } = useAuth();
  const userID = user?.uid;
  const [userdata, setUserdata] = useState();
  const [patient ,setPatient] = useState();

  const fetchUserData = async () => {
    try {
      if (userID) {
        const userRef = doc(db, "users", userID);
        const docSnap = await getDoc(userRef);
        if (docSnap.exists()) {
          const userdata = docSnap.data();
          setUserdata(userdata);
        } else {
          console.log("No such document!");
        }
      }
    } catch (error) {
      console.error("Error fetching user data:", error);
    }
  };
  useEffect(() => {
    fetchUserData();
  }, [userID]);

  // const updateUser = async (patientdata) => {
  //     try {
  //       const userRef = doc(db, 'users', userID);
  //       await updateDoc(userRef, {
  //         name: patientdata.name,
  //         email: patientdata.email,
  //         address: patientdata.address,
  //         nic: patientdata.nic,
  //         mobile: patientdata.mobile,
  //         birthday: patientdata.birthday,
  //         profilepic: patientdata.image,

  //         securityid:patientdata.securityid,
  //         isUpdate: true

  //       });
  //       console.log('User profile updated successfully');
  //       alert('User profile updated successfully')
  //       window.location.reload();
  //     } catch (error) {
  //       console.error('Error updating user profile:', error);
  //       alert('Error updating user profile:', error)
  //     }
  //   }

  const updateUser = async (patientdata) => {
    try {
      const reportsCollection = collection(db, "reports");
      const queryRef = query(reportsCollection, where("userID", "==", userID));
      const reportSnapshot = await getDocs(queryRef);

      const updatereports = reportSnapshot.docs.map((docSnapshot) => {
        const reportRef = doc(db, "reports", docSnapshot.id);
        return updateDoc(reportRef, {
          nic: patientdata.nic,
          
          securityid: patientdata.securityid,
        });
      });

      await Promise.all(updatereports);

      const userRef = doc(db, "users", userID);
      await updateDoc(userRef, {
        name: patientdata.name,
        email: patientdata.email,
        reminder:patientdata.reminder
      });

      console.log("User profile and reports updated successfully");
      alert("User profile and reports updated successfully");
      window.location.reload();
    } catch (error) {
      console.error("Error updating user profile and reports:", error);
      alert("Error updating user profile and reports:", error);
    }
  };

  const updateDoctor = async (doctorInfo) => {
    try {
      const userRef = doc(db, "users", userID);
      await updateDoc(userRef, {
        name: doctorInfo.name,
        email: doctorInfo.email,
        address: doctorInfo.address,
        nic: doctorInfo.nic,
        mobile: doctorInfo.mobile,
        registernumber: doctorInfo.registernumber,
        profilepic: doctorInfo.image,
      });
      console.log("Doctor profile updated successfully");
      alert("Doctor profile updated successfully");
      window.location.reload();
    } catch (error) {
      console.error("Error updating doctor profile:", error);
      alert("Error updating doctor profile:", error);
    }
  };

  return (
    <ProfileContext.Provider value={{ updateUser, userdata, updateDoctor ,patient,setPatient }}>
      {children}
    </ProfileContext.Provider>
  );
};
export const useProfile = () => {
  return useContext(ProfileContext);
};
