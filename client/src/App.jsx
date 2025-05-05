import { useState } from 'react';
import './App.css';
import logo from './assets/PANYNJ_logo.png';

function App() {
  const [query, setQuery] = useState('');
  const [chatLog, setChatLog] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setChatLog(prev => [...prev, { type: 'user', text: query }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ query })
      });

      if (!response.ok) throw new Error('Failed to get response from server.');

      const data = await response.json();

      setChatLog(prev => [...prev, { type: 'bot', text: data.response }]);
    } catch (error) {
      console.error(error);
      setChatLog(prev => [...prev, { type: 'bot', text: 'Error contacting server.' }]);
    }

    setIsLoading(false);
    setQuery('');
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if(!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try{
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('File upload failed.');

      const data = await response.json();
      alert(`File uploaded successfully: ${data.status}`);
    } catch (error) {
      console.error(error);
      alert('Error uploading file.');
    }

    setSelectedFile(null);
    };

    return (
      <main className="main-wrapper">
  <div className="chat-container">
    <header className="chat-header">
     <img src={logo} alt="PANYNJ Logo" className="chat-logo" />
    </header>


    <div className="upload-form">
      <form onSubmit={handleFileUpload}>
        <input
          type="file"
          onChange={(e) => setSelectedFile(e.target.files[0])}
        />
        <button type="submit">Upload</button>
      </form>
    </div>

    <div className="chat-window">
      {chatLog.map((entry, idx) => (
        <div key={idx} className={`chat-message ${entry.type}`}>
          <div className="message-bubble">
            <strong>{entry.type === 'user' ? 'You' : 'Bot'}:</strong> {entry.text}
          </div>
        </div>
      ))}
      {isLoading && <div className="chat-message bot loading">Bot is thinking...</div>}
    </div>

    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Ask a legal question..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={isLoading}
      />
      <button type="submit" disabled={isLoading}>Send</button>
    </form>
  </div>
</main>

    );  
}

export default App;
