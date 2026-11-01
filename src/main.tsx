import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { isDesktop } from "./platform";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// The web version works offline once installed. The desktop app serves its
// files from inside the binary and needs none of this.
if (import.meta.env.PROD && !isDesktop() && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch((err) => {
      console.warn("[Nabu] Offline support unavailable:", err);
    });
  });
}
