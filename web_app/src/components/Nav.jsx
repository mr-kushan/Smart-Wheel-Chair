import { React, useState, useEffect } from "react";
import { AiOutlineClose, AiOutlineMenu } from "react-icons/ai";
import { useNavigate } from "react-router-dom";
// import { auth } from "../../../firebase";
// import { useAuth } from "../../Context/AuthContext";
import { Link } from "react-router-dom";
// import { useFunction } from "../../Context/functionOne";
import axios from "axios";
// import { ref, uploadString, getStorage } from "firebase/storage";
import { ProfileContextProvider, useProfile } from "./Context/ProfileContext";

const Nav = ({ isOpen, onClose }) => {
  // const { user, logOut } = useAuth();
  const navigate = useNavigate();
  const [userr, setUser] = useState(null);
  const {patient} = useProfile()
  const [name , setName] = useState("")
  useEffect(() => {
    const savedPatient = localStorage.getItem("patient");
    if (savedPatient) {
      const parsedPatient = JSON.parse(savedPatient);
      setName(parsedPatient.name); // Set name state
      console.log("Loaded patient from storage:", parsedPatient);
    }
  }, []);
  
  const handleLogout = async (event) => {
    event.preventDefault();
    try {
      // await logOut();
      console.log("logout success");
      navigate("/");
    } catch (handledError) {
      console.log(handledError.message);
    }
  };



  return (
    <div
  className={`w-[300px] fixed -ml-96 lg:ml-0 md:ml-0 sm:ml-0 left-0 top-0 flex flex-col justify-between h-screen px-16 pb-3 ${
    isOpen ? "z-50 ml-2" : "z-0"
  }`}
  style={{
    backgroundImage: "radial-gradient(circle, #D4E89D, #D4E89D,#D4E89D )",// Adjust colors as needed
  }}
>
      <div>
        <div className="sm:hidden right-0 p-4 absolute text-white cursor-pointer">
          <AiOutlineClose onClick={onClose} />
        </div>
        <div className="mt-8 text-center">
          <img
            src="https://flowbite.com/docs/images/people/profile-picture-5.jpg"
            alt=""
            className="m-auto h-28 w-28 rounded-full object-cover lg:h-28 lg:w-28"
          />
          <h5 className="mt-4  text-xl font-semibold text-gray-600 lg:block dark:text-gray-300">
            {name}
          </h5>
          <span className=" text-gray-400 lg:block">Care Giver</span>
            {/* {console.log(patient.name)} */}
        </div>

        <ul className="mt-8 space-y-2 tracking-wide">
          <li>
            <a
              href="/admin"
              aria-label="dashboard"
              className="relative flex items-center space-x-4 rounded-xl hover:bg-gradient-to-r from-sky-600 to-cyan-400 px-4 py-3 text-white"
            >
              <span className="-mr-1 font-medium">Function 1</span>
            </a>
          </li>
          <li>
            <a
              href="/function02"
              className="group flex items-center space-x-4 rounded-md px-4 py-3 text-gray-600 dark:text-gray-300"
            >
              <span className="group-hover:text-gray-700 dark:group-hover:text-gray-50">
                Set Reminders
              </span>
            </a>
          </li>
          <li>
            <a
              href="/sentreminders"
              className="group flex items-center space-x-4 rounded-md px-4 py-3 text-gray-600 dark:text-gray-300"
            >
              <span className="group-hover:text-gray-700 dark:group-hover:text-gray-50">
                Sent Reminders 
              </span>
            </a>
          </li>
          <li>
            <a
              href="/function03"
              className="group flex items-center space-x-4 rounded-md px-4 py-3 text-gray-600 dark:text-gray-300"
            >
              <span className="group-hover:text-gray-700 dark:group-hover:text-gray-50">
                Check Heart
              </span>
            </a>
          </li>
          <li>
            <a
              href="/function04"
              className="group flex items-center space-x-4 rounded-md px-4 py-3 text-gray-600 dark:text-gray-300"
            >
              <span className="group-hover:text-gray-700 dark:group-hover:text-gray-50">
                Plan Generator
              </span>
            </a>
          </li>

          <li>
            <button
              className="group flex items-center space-x-4 rounded-md px-4 py-3 text-gray-600 dark:text-gray-300"
              onClick={handleLogout}
            >
              <span className="-mr-1 font-medium">Sign Out</span>
            </button>
          </li>
        </ul>
      </div>

      <div className="  items-center justify-between  ">
        <p className="text-green-500 text-2xl ">
          <span className="font-bold">Smart</span> Wheeler
        </p>
      </div>
    </div>
  );
};

export default Nav;
