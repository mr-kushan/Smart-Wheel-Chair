import React, { useState } from 'react';
import { useAuth } from "../Context/AuthContext";
import { useNavigate } from "react-router-dom";
// import { useAuth } from '../Context/AuthContext';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [lastName, setLastName] = useState('');
  const [controllerId, setControllerId] = useState('');
  // const [confirmPassword, setConfirmPassword] = useState('');
  const { signIn, signupError } = useAuth();



  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    const CareGiverInfo = {
      password: password,
      email: email,

    };
  
    try {
      console.log(email, password, lastName, controllerId);
      await signIn(CareGiverInfo);
  
      // Sign in the user after successful account creation
      await signIn(email, password); // Assuming `createPatient` is your login method
  
      // navigate('/nav'); // Redirect to a protected route after login
  
      if (signupError) {
        alert(signupError);
      }
    } catch (error) {
      console.log('error', error);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gradient-to-r from-[#e0f2e9] to-[#d1f7c4]">
      <div className="flex flex-row items-center max-w-5xl p-8 gap-10">
        {/* Left section for title and description */}
        <div className="max-w-md text-left">
          <h1 className="text-4xl font-bold text-teal-800">Smart Wheeler</h1>
          <p className="text-md text-gray-700 mt-2">
            Welcome to Smart Wheelchair Controller – Empowering Independence Through Seamless Mobility Control.
          </p>
        </div>

        {/* Right section for the sign-up form */}
        <div className="bg-white p-8 rounded-xl shadow-lg max-w-md w-full  ">
          <h1 className="text-2xl font-semibold text-teal-600 mb-4">LOGIN</h1>
          <form onSubmit={handleSubmit} className="flex flex-col">
            

            <label className="text-sm font-medium text-gray-600 mb-1" htmlFor="email">Email Address *</label>
            <input
              type="email"
              id="email"
              placeholder="Enter your email address"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="p-3 border border-gray-300 rounded focus:outline-none focus:border-teal-500 mb-4"
            />

      
            <label className="text-sm font-medium text-gray-600 mb-1" htmlFor="password">Password *</label>
            <input
              type="password"
              id="password"
              placeholder="Enter your password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="p-3 border border-gray-300 rounded focus:outline-none focus:border-teal-500 mb-4"
            />
         

            <button
              type="submit"
              className="bg-teal-600 text-white py-3 rounded hover:bg-teal-700 transition duration-300"
            >
              Login
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Login;