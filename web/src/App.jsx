import { useState } from "react";

import Chat from "./features/Chat.jsx";
import Search from "./features/Search.jsx";
import Classify from "./features/Classify.jsx";
import Detect from "./features/Detect.jsx";

export default function App() {
  const [page, setPage] = useState("chat");

  return (
    <main className="app">
      <h1>AI Web Apps</h1>

      <nav className="main-nav">
        <button
          type="button"
          className={page === "chat" ? "nav-button active" : "nav-button"}
          onClick={() => setPage("chat")}
        >
          RAG Chatbot
        </button>

        <button
          type="button"
          className={page === "search" ? "nav-button active" : "nav-button"}
          onClick={() => setPage("search")}
        >
          Image Retrieval
        </button>

        <button
          type="button"
          className={page === "classify" ? "nav-button active" : "nav-button"}
          onClick={() => setPage("classify")}
        >
          Image Classification
        </button>

        <button
          type="button"
          className={page === "detect" ? "nav-button active" : "nav-button"}
          onClick={() => setPage("detect")}
        >
          Object Detection
        </button>
      </nav>

      {page === "chat" && <Chat />}

      {page === "search" && <Search />}

      {page === "classify" && <Classify />}

      {page === "detect" && <Detect />}
    </main>
  );
}
