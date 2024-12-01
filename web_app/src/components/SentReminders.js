import React, { useEffect, useState } from 'react';
import { collection, query, where, getDocs } from 'firebase/firestore';
import { db } from '../firebase';
import Nav from './Nav';

const SentReminder = () => {
  const [menu, setMenu] = useState(false);
  const [reminders, setReminders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [patientId, setPatientId] = useState('');

  useEffect(() => {
    const savedPatient = localStorage.getItem('patient');
    if (savedPatient) {
      const parsedPatient = JSON.parse(savedPatient);
      setPatientId(parsedPatient.id);
      console.log('Loaded patient ID from storage:', parsedPatient.id);
    }
  }, []);

  useEffect(() => {
    const fetchReminders = async () => {
      try {
        setLoading(true);
        setError('');

        if (!patientId) {
          throw new Error('Patient ID is required to fetch reminders.');
        }

        const usersCollection = collection(db, 'users');
        const queryRef = query(usersCollection, where('id', '==', patientId));
        const userSnapshot = await getDocs(queryRef);

        if (userSnapshot.empty) {
          throw new Error(`No user found with ID "${patientId}".`);
        }

        const userData = userSnapshot.docs[0].data();
        const fetchedReminders = userData.reminders || [];

        setReminders(fetchedReminders);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (patientId) {
      fetchReminders();
    }
  }, [patientId]);

  return (
    <div className="h-screen flex">
      {/* Sidebar */}
      <aside
        className={`w-64 bg-white shadow-md fixed inset-y-0 left-0 z-20 transform transition-transform ${
          menu ? 'translate-x-0' : '-translate-x-full'
        } sm:translate-x-0`}
      >
        <Nav isOpen={menu} onClose={() => setMenu(false)} />
      </aside>

      {/* Main Content */}
      <div className="flex-1 bg-gray-100 p-6 ml-64 overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Sent Reminders</h2>
        {loading ? (
          <p className="text-gray-500">Loading reminders...</p>
        ) : error ? (
          <p className="text-red-500">Error: {error}</p>
        ) : reminders.length > 0 ? (
          <ul className="space-y-4">
            {reminders.map((reminder, index) => (
              <li key={index} className="p-4 border rounded shadow">
                <p>
                  <strong>Date:</strong> {reminder.date || 'N/A'}
                </p>
                <p>
                  <strong>Time:</strong> {reminder.time || 'N/A'}
                </p>
                <p>
                  <strong>Description:</strong> {reminder.description || 'N/A'}
                </p>
                <p>
                  <strong>Patient ID:</strong> {reminder.patientID || 'N/A'}
                </p>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500">No reminders found for this patient.</p>
        )}
      </div>
    </div>
  );
};

export default SentReminder;
