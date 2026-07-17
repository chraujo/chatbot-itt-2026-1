import { useState } from "react";
import Login from "./Login";
import Chat from "./Chat";
import { ThemeProvider, useTheme } from "./ThemeContext";

function AppContent() {
  const [idToken, setIdToken] = useState<string | null>(null);
  const [email, setEmail] = useState<string | null>(null);
  const { darkMode, toggleTheme } = useTheme();

  const handleLogin = (token: string, email: string) => {
    setIdToken(token);
    setEmail(email);
  };

  const logout = () => {
    setIdToken(null);
    setEmail(null);
  };

  return (
    <div style={{...appStyle, background: darkMode ? '#0c0c0f' : '#f9fafb'}}>
      {/* Theme Toggle Button - Fixed Position */}
      <button
        onClick={toggleTheme}
        style={themeToggleButtonFixed}
        title="Alternar tema"
      >
        <img
          src={darkMode ? '/sun.svg' : '/moon.svg'}
          alt="Toggle theme"
          style={{...themeIconStyle, filter: darkMode ? 'brightness(0) invert(1)' : 'brightness(0.6)'}}
        />
      </button>

      {!idToken || !email ? (
        <Login onLogin={handleLogin} />
      ) : (
        <Chat idToken={idToken} email={email} onLogout={logout} />
      )}
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

const appStyle = {
  background: "#0c0c0f",
  minHeight: "100vh",
  color: "#e5e7eb",
  position: "relative" as const,
};

const themeToggleButtonFixed = {
  position: "fixed" as const,
  top: "20px",
  right: "20px",
  background: "rgba(255, 255, 255, 0.1)",
  border: "1px solid rgba(255, 255, 255, 0.2)",
  borderRadius: "8px",
  padding: "10px",
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 9999,
  transition: "all 0.2s",
  backdropFilter: "blur(10px)",
};

const themeIconStyle = {
  width: "24px",
  height: "24px",
  stroke: "#e5e7eb",
};
