// main.jsx
// ─────────────────────────────────────────────────────────────────────────────
// This is the entry point of the entire React application.
// It does one job: find the <div id="root"> in index.html and mount our App
// component inside it. Everything else flows from App.jsx.
// ─────────────────────────────────────────────────────────────────────────────

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";

// StrictMode is a development helper — it intentionally double-renders
// components in dev to catch bugs early. It has zero effect in production.
createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
