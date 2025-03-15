import React, { useState } from 'react';
import { useAuth } from "../Context/AuthContext";
import { useNavigate } from "react-router-dom";

const SignUp = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [lastName, setLastName] = useState('');
  const [controllerId, setControllerId] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const { createCareGiver, createPatient ,createPatients,signupError} = useAuth();
  const [userType,setUserType] =useState('')
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
    if(userType === "careGiver"){
      
    const CareGiverInfo = {
      name: lastName,
      email: email,
      userType:userType,
      controllerID: controllerId,
      password: password,
    };
    console.log(email, password, lastName, controllerId);
    await createCareGiver(CareGiverInfo);
    }else{
      const patientInfo = {
        name: lastName,
        email: email,
        userType:"patient",
        controllerID: controllerId,
        password: password,
        
      }
      await createPatients(patientInfo); 
    }
      // Sign in the user after successful account creation
  
      navigate('/login'); // Redirect to a protected route after login
  
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
          <h1 className="text-2xl font-semibold text-teal-600 mb-4">Sign Up</h1>
          <form onSubmit={handleSubmit} className="flex flex-col">
          <div className="flex items-center gap-2">
                  
                  <select
                    onChange={(e) => setUserType(e.target.value)}
                    value={userType}
                    className="w-full px-5 py-3 rounded-lg font-medium bg-gray-100 border border-gray-200 placeholder-gray-500 text-sm focus:outline-none focus:border-gray-400 focus:bg-white"
                    name="userType"
                    id="userType"
                  >
                    <option value="patient">Patient</option>
                    <option value="careGiver">CareGiver</option>
                  </select>
                </div>
            <label className="text-sm font-medium text-gray-600 mb-1" htmlFor="lastName">Last Name *</label>
            <input
              type="text"
              id="lastName"
              placeholder="Enter your last name"
              required
              value={lastName}
              onChange={(e) => setLastName(e.target.value)}
              className="p-3 border border-gray-300 rounded focus:outline-none focus:border-teal-500 mb-4"
            />

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

            <label className="text-sm font-medium text-gray-600 mb-1" htmlFor="controllerId">Controller ID *</label>
            <input
              type="text"
              id="controllerId"
              placeholder="Enter your controller id"
              required
              value={controllerId}
              onChange={(e) => setControllerId(e.target.value)}
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
            
            <label className="text-sm font-medium text-gray-600 mb-1" htmlFor="confirmPassword">Confirm Password *</label>
            <input
              type="password"
              id="confirmPassword"
              placeholder="Re-enter your password"
              required
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="p-3 border border-gray-300 rounded focus:outline-none focus:border-teal-500 mb-6"
            />

            <button
              type="submit"
              className="bg-teal-600 text-white py-3 rounded hover:bg-teal-700 transition duration-300"
            >
              Sign Up
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default SignUp;
