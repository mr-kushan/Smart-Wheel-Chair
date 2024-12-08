import React, { useEffect, useState } from 'react';
import Nav from './Nav';
import { useProfile } from './Context/ProfileContext';
import { collection, query, where, getDocs, doc, updateDoc } from 'firebase/firestore'
import { db } from '../firebase';
const Function02 = () => {
  const [menu, setMenu] = useState(false); // Use boolean for `menu` state
  const { patient,updateUser } = useProfile();
  const [name, setName] = useState('');
  const [patientID, setPatientID] = useState('');
  const [date, setDate] = useState('');
  const [time, setTime] = useState('');
  const [description, setDescription] = useState('');

  useEffect(() => {
    const savedPatient = localStorage.getItem("patient");
    if (savedPatient) {
      const parsedPatient = JSON.parse(savedPatient);
      setName(parsedPatient.id); // Set name state
      console.log("Loaded patient from storage:", parsedPatient);
    }
  }, []);
  

  const handleSave = async () => {
    const newReminder = { patientID, date, time, description };
    console.log('Reminder details:', newReminder);
  
    try {
      // Ensure patientID is provided
      if (!patientID) {
        console.error('Patient ID is required');
        return;
      }
  
      // Reference the user document by patientID
      const userRef = doc(db, 'users', patientID);
  
      // Fetch the existing user document
      const userSnapshot = await getDocs(query(collection(db, 'users'), where('id', '==', patientID)));
  
      if (userSnapshot.empty) {
        console.error(`User with Patient ID "${patientID}" not found`);
        return;
      }
  
      // Get the current reminders array or default to an empty array
      const userData = userSnapshot.docs[0].data();
      const currentReminders = userData.reminders || [];
  
      // Add the new reminder to the array
      const updatedReminders = [...currentReminders, newReminder];
  
      // Update the user document with the new array of reminders
      await updateDoc(userRef, { reminders: updatedReminders });
  
      console.log('User reminders updated successfully');
    } catch (error) {
      console.error('Error updating user reminders:', error);
    }
  };
  const toggleMenu = () => {
    setMenu(!menu);
  };

  return (
    <div className="bg-gradient-to-white to-white/80 from-white-950 bg-white-900 w-full h-screen flex relative sm:overflow-x-hidden">
      {/* Navigation Sidebar */}
      <div
        className={`transition-transform transform ${
          menu ? 'translate-x-0' : '-translate-x-full'
        } sm:translate-x-0 duration-500 fixed sm:relative z-10 w-64 bg-white shadow-md h-full`}
      >
        <Nav isOpen={menu} onClose={() => setMenu(false)} />
      </div>

      {/* Main Content */}
      <div className={`flex-1 flex flex-col md:flex-row items-start justify-between p-8 space-y-8 md:space-y-0 md:space-x-8 h-full ${menu ? 'ml-64' : ''}`}>
        {/* Toggle Button for Nav in Mobile View */}
        <button
          onClick={toggleMenu}
          className="sm:hidden bg-teal-500 text-white px-4 py-2 rounded shadow mb-4"
        >
          Toggle Menu
        </button>

        {/* Left Section: Reminder Form */}
        <div className="w-full md:w-1/2 flex flex-col justify-center space-y-6">
          <h2 className="text-2xl font-semibold mb-4">Set Reminders By Voice</h2>
          <div className="flex justify-center">
            <button className="flex items-center justify-center px-2 py-1 bg-teal-500 text-white rounded-xl shadow hover:bg-teal-600 focus:outline-none text-xs w-40">
              <span>SPEAK</span>
              <span className="ml-1 text-xs">🎤</span> {/* Replace with microphone icon if available */}
            </button>
          </div>

          <div className="mt-10">
            <h3 className="text-xl font-semibold mb-4">Set Reminders</h3>
            <form className="space-y-6" onSubmit={(e) => e.preventDefault()}>
              <div>
                <label className="block font-medium text-gray-700 mb-2" htmlFor="patientId">
                  Patient ID
                </label>
                <input
                  type="text"
                  id="patientId"
                  placeholder="Enter your Patient ID"
                  onChange={(e) => setPatientID(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-teal-500"
                />
              </div>

              <div>
                <label className="block font-medium text-gray-700 mb-2">Date and Time</label>
                <div className="flex space-x-4">
                  <input
                    type="time"
                    placeholder="Start Time"
                    onChange={(e) => setTime(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-teal-500"
                  />
                  <input
                    type="date"
                    placeholder="Start Date"
                    onChange={(e) => setDate(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-teal-500"
                  />
                </div>
              </div>

              <div>
                <label className="block font-medium text-gray-700 mb-2" htmlFor="description">
                  Description
                </label>
                <textarea
                  id="description"
                  placeholder="Enter your Description"
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-teal-500"
                  rows="4"
                ></textarea>
              </div>

              <div className="flex justify-center">
                <button
                  onClick={handleSave}
                  className="flex items-center justify-center px-2 py-1 bg-teal-500 text-white rounded-xl shadow hover:bg-teal-600 focus:outline-none text-xs w-40"
                >
                  <span>SAVE</span>
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Right Section: Map */}
        <div className="w-full md:w-1/2 flex flex-col items-center h-full">
          <div className="text-center mb-6">
            <h2 className="text-xl font-semibold">Find My Wheeler</h2>
            <button className="text-gray-700 underline">Trace Location</button>
          </div>
          <div className="w-full h-full bg-gray-300 rounded shadow-md flex items-center justify-center">
            {/* Replace with actual map embed or map component */}
            <p className="text-gray-600">Map goes here</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Function02;
