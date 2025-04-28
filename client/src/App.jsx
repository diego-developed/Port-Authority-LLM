import { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [chatLog, setChatLog] = useState([]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setChatLog(prev => [...prev, { type: 'user', text: query }]);

    // TODO: Send query to backend
    setTimeout(() => {
      setChatLog(prev => [...prev, { type: 'bot', text: 'Placeholder response from legal bot.' }]);
    }, 500);

    setQuery('');
  };

  return (
    <div className="dashboard">
      <div className="sidebar">
        <h2>🗂️ Conversations</h2>
        {/* TODO: List past chats */}
        <p className="conversation-placeholder">No past conversations</p>
      </div>

      <div className="app-container">
        <header className="header">
          <h1>📄 Legal Document Chatbot</h1>
        </header>

        <div className="upload-section">
          <label htmlFor="docUpload">📎 Upload Document:</label>
          <input type="file" id="docUpload" name="docUpload" />
          {/* TODO: Send document to backend */}
        </div>

        <div className="chat-box">
          {chatLog.map((entry, idx) => (
            <div key={idx} className={`chat-entry ${entry.type}`}>
              <strong>{entry.type === 'user' ? 'You' : 'Bot'}:</strong> {entry.text}
            </div>
          ))}
        </div>

        <form className="input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Ask a legal question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit">Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;
