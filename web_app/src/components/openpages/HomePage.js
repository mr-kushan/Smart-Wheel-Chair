import React from "react";
import homeImage from "../../assets/homeImage.jpg";
import { useNavigate } from "react-router-dom";
const HomePage = () => {
  const navigate = useNavigate();
  const handleLogin =async() => {
    navigate("/login")
  }
  const handleSignup = async() => {
    navigate("/signup")
  }
  return (
    <div className="background">
      <div className="overlay">
        <h1 className="title">Smart Wheeler</h1>
        <div className="buttonContainer">
          <button onClick={handleSignup} className="button signup" >Sign Up</button>
          <button className="button login" onClick={handleLogin}>Log In</button>
        </div>
      </div>
    </div>
  );
};

const styles = {
  background: {
    backgroundImage: `url(${homeImage})`, // Replace with the actual image path or URL
    backgroundSize: "cover",
    backgroundPosition: "center",
    height: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  overlay: {
    textAlign: "center",
    // color: "white",
    // backgroundColor: "rgba(0, 0, 0, 0.5)", // Dark overlay for text readability
    padding: "20px",
    borderRadius: "10px",
  },
  title: {
    fontSize: "4em",
    fontWeight: "bold",
    marginBottom: "20px",
    fontFamily: "'Lobster', cursive",
    color: "white",  

  },
  buttonContainer: {
    display: "flex",
    gap: "20px",
    justifyContent: "center",
  },
  button: {
    padding: "10px 20px",
    fontSize: "0.7em",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    transition: "background-color 0.3s",
    color: "white",
  },
  signup: {
    backgroundColor: "#2D9596",
  },
  login: {
    backgroundColor: "#ECF4D6",
    color: "#2D9596"
  },
};

// Applying the styles object to the document
Object.keys(styles).forEach((className) => {
  const style = document.createElement("style");
  style.innerHTML = `.${className} { ${Object.entries(styles[className])
    .map(([key, value]) => `${camelToKebab(key)}: ${value};`)
    .join(" ")} }`;
  document.head.appendChild(style);
});

// Helper function to convert camelCase to kebab-case
function camelToKebab(str) {
  return str.replace(/([a-z])([A-Z])/g, "$1-$2").toLowerCase();
}

export default HomePage;
