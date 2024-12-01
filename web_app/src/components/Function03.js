import React, { useState } from 'react';
import Nav from './Nav';

const Function03 = () => {
  const [menu, setMenu] = useState(false); // Use boolean for `menu` state

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
          className="sm:hidden bg-2D9596-500 text-white px-4 py-2 rounded shadow mb-4 "
        >
          Toggle Menu
        </button>

        {/* Left Section: Reminder Form */}
        <div className="w-full md:w-1/2 flex flex-col justify-center space-y-6">
          


          <div className="mt-10">
            <h3 className="text-2xl font-semibold mb-4">Sensor Data</h3>
            <form className="space-y-6">
              <div>
                <label className="block font-medium  text-gray-700 mb-2" htmlFor="heartRate">
                  Heart Rate
                </label>
                <input
                  type="text"
                  id="heartRate"
                  placeholder="Enter Patient Heart Rate"
                  className="w-80 px-4 py-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-teal-500 "
                />
              </div>
              <div>
                <label className="block font-medium  text-gray-700 mb-2" htmlFor="oxygentLevels">
                  Oxygen Levels
                </label>
                <input
                  type="text"
                  id="oxygentLevels"
                  placeholder="Enter Patient Oxygen Levels"
                  className="w-80 px-4 py-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-teal-500 "
                />
              </div>
              <div>
                <label className="block font-medium  text-gray-700 mb-2" htmlFor="bloodPressure">
                  Blood Pressure
                </label>
                <input
                  type="text"
                  id="bloodPressure"
                  placeholder="Enter Patient Blood Pressure"
                  className="w-80 px-4 py-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-teal-500 "
                />
              </div>
              <div>
                <label className="block font-medium  text-gray-700 mb-2" htmlFor="bodyTemperature">
                  Body Temperature
                </label>
                <input
                  type="text"
                  id="bodyTemperature"
                  placeholder="Enter Patient Body Temperature"
                  className="w-80 px-4 py-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-teal-500 "
                />
              </div>
              <div className="flex justify-center">
            <button className="flex items-center justify-center px-4 py-2 bg-teal-500 text-white rounded-xl shadow hover:bg-teal-600 focus:outline-none text-xs w-40">
    <span>SAVE</span>
    
  </button>
</div>

            </form>
          </div>
        </div>

        {/* Right Section: Map */}
        <div className="w-full md:w-1/2 flex flex-col items-center h-full">
          <div className="text-center mb-6">
            <h2 className="text-xl font-semibold">The Result</h2>
          </div>
          <div className="w-full h-full bg-gray-300 rounded shadow-md flex items-center justify-center">
            {/* Replace with actual map embed or map component */}
            <p className="text-gray-600">Result goes here</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Function03;
