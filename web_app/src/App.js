import './App.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from './components/openpages/HomePage';
import SignUp from './components/openpages/SignUp';
import Login from './components/openpages/LogIn';
import Nav from './components/Nav';
import Function02 from './components/Function02';
import Function03 from './components/Function03';
import Function04 from './components/Function04';
import PatientGrid from './components/PatientGrid';
import SentReminders from './components/SentReminders';
import AuthContextProvider from './components/Context/AuthContext';
import {ProfileContextProvider} from './components/Context/ProfileContext';

function App() {
  return (
    <Router>
      <AuthContextProvider>
        <ProfileContextProvider>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/signup" element={<SignUp />} />
            <Route path="/login" element={<Login />} />
            <Route path="/nav" element={<Nav />} />
            <Route path="/function02" element={<Function02 />} />
            <Route path="/function03" element={<Function03 />} />
            <Route path="/function04" element={<Function04 />} />
            <Route path="/patientgrid" element={<PatientGrid />} />
            <Route path="/sentreminders" element={<SentReminders />} />
            
          </Routes>
        </ProfileContextProvider>
      </AuthContextProvider>
    </Router>
  );
}

export default App;
