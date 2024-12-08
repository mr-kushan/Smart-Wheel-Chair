import React, { useState, useEffect, Children } from 'react';
import { FiBell } from 'react-icons/fi';
import { doc, getDocs, collection, deleteDoc, setDoc } from "firebase/firestore";
import { db } from "../firebase"; // Ensure this points to your Firebase config file
import { useAuth } from "./Context/AuthContext";
import { useProfile } from "./Context/ProfileContext";
// import { Navigate } from 'react-router-dom';
import { useNavigate } from "react-router-dom";
import {App} from "../App"

const PatientGrid = () => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [pendingRequests, setPendingRequests] = useState([]);
  const [addToGrid, setAddToGrid] = useState([]);
  const [controllerID, setControllerID] = useState("");
  const { updateUser, userdata,patient,setPatient } = useProfile();
  const { logOut } = useAuth();
  const navigate = useNavigate();

  // Toggle notification display
  const handleBellClick = () => {
    setShowNotifications(!showNotifications);
  };

  // Set the controller ID when userdata changes
  useEffect(() => {
    if (userdata && userdata.controllerID) {
      setControllerID(userdata.controllerID);
      console.log("Controller ID:", userdata.controllerID);
    }
  }, [userdata]);

  // Fetch pending accounts from "newUsers" collection
  const fetchPendingRequests = async () => {
    try {
      const querySnapshot = await getDocs(collection(db, "newUsers"));
      const requests = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data();
        console.log(`Controller ID: ${data.controllerID}, User Type: ${data.userType}`);
        if (data.controllerID === controllerID && data.userType === "patient") {
          requests.push({ id: doc.id, ...data });
        }
      });
      setPendingRequests(requests);
    } catch (error) {
      console.error("Error fetching pending requests:", error);
    }
  };

  // Fetch users to add to the grid
  const addToPatientGrid = async () => {
    try {
      const querySnapshot = await getDocs(collection(db, "users"));
      const patients = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data();
        if (data.controllerID === controllerID) {
          patients.push({ id: doc.id, ...data });
        }
      });
      setAddToGrid(patients);
    } catch (error) {
      console.error("Error fetching users for the grid:", error);
    }
  };

  // Handle account acceptance
  const handleAccept = async (request) => {
    try {
      const userRef = doc(db, "users", request.id);
      await setDoc(userRef, request); // Add user to "users" collection
      await deleteDoc(doc(db, "newUsers", request.id)); // Remove from "newUsers" collection
      setPendingRequests((prev) => prev.filter((item) => item.id !== request.id));
      console.log("Account accepted and moved to 'users' collection.");
    } catch (error) {
      console.error("Error accepting account:", error);
    }
  };

  // Handle account rejection
  const handleDecline = async (requestID) => {
    try {
      await deleteDoc(doc(db, "newUsers", requestID)); // Remove from "newUsers"
      setPendingRequests((prev) => prev.filter((item) => item.id !== requestID));
      console.log("Account declined and removed from 'newUsers' collection.");
    } catch (error) {
      console.error("Error declining account:", error);
    }
  };

  const handleGo = (patient) =>{
    console.log(patient.name)
    setPatient(patient)
    localStorage.setItem("patient", JSON.stringify(patient));

    navigate("/nav")

  }
  // Initial data fetch
  useEffect(() => {
    fetchPendingRequests();
    addToPatientGrid();
  }, [controllerID]);


  return (
    <div className="relative p-6">
      <h1 className="text-2xl font-semibold mb-4 text-center">Smart Wheeler Functions</h1>

      {/* Notification bell and logout button */}
      <div className="absolute top-4 right-4 flex space-x-4">
        <button
          onClick={handleBellClick}
          className="p-2 bg-teal-500 text-white rounded-full shadow hover:bg-teal-600"
        >
          <FiBell size={24} />
        </button>
        <button
          onClick={logOut} // Ensure you handle the logout properly
          className="px-4 py-2 bg-teal-500 text-white rounded-lg shadow hover:bg-teal-600"
        >
          Logout
        </button>
      </div>

      <div className="grid gap-4">
        {addToGrid.length > 0 ? (
          addToGrid.map((patient) => (
            <div
              key={patient.id}
              className="bg-white p-4 rounded-lg shadow-md flex justify-between items-center"
            >
              <div className="flex items-center space-x-4">
                <div className="w-16 h-16 bg-gray-300 rounded-full flex justify-center items-center">
                  <span>Image</span>
                </div>
                <div>
                  <h2 className="font-bold">{patient.name || "Patient Name"}</h2>
                  {/* <app userInfo={patient.name}/> */}
                  <p className="text-gray-600">{patient.description || "Description"}</p>
                </div>
              </div>
              <button onClick={() =>handleGo(patient)} className="w-10 h-10 bg-teal-500 text-white rounded-full shadow hover:bg-teal-600">
                → 
              </button>
            </div>
          ))
        ) : (
          <p className="text-gray-700 text-center">No data to display.</p>
        )}
      </div>

      {/* Notification Panel */}
      {showNotifications && (
        <div className="absolute top-16 right-4 bg-white p-4 rounded-lg shadow-lg w-64">
          <h2 className="text-lg font-semibold mb-2">Pending Requests</h2>
          <ul>
            {pendingRequests.length > 0 ? (
              pendingRequests.map((request) => (
                <li key={request.id} className="flex justify-between items-center mb-2">
                  <span>{request.name}</span>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleAccept(request)}
                      className="px-2 py-1 bg-green-500 text-white rounded"
                    >
                      Accept
                    </button>
                    <button
                      onClick={() => handleDecline(request.id)}
                      className="px-2 py-1 bg-red-500 text-white rounded"
                    >
                      Decline
                    </button>
                  </div>
                </li>
              ))
            ) : (
              <p className="text-gray-700">No pending requests.</p>
            )}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PatientGrid;
