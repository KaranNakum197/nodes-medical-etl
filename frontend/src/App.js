import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

function App() {
  const [records, setRecords] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  // Fetch existing records on mount
  useEffect(() => {
    fetchRecords();
  }, []);

  async function fetchRecords() {
    try {
      const res = await axios.get(`${API_BASE}/records`);
      setRecords(res.data);
    } catch (err) {
      console.error('Failed to fetch records:', err);
    }
  }

  async function handleUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setMessage('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_BASE}/extract`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMessage(`Extracted successfully: ${res.data.filename}`);
      fetchRecords();
    } catch (err) {
      setMessage(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 900, margin: '2rem auto', padding: '0 1rem' }}>
      <h1>🏥 Medical ETL Dashboard</h1>
      <p>Upload a medical document (PDF or image) to extract structured data.</p>

      <div style={{ marginBottom: '1.5rem' }}>
        <input
          type="file"
          accept="application/pdf,image/png,image/jpeg"
          onChange={handleUpload}
          disabled={uploading}
        />
        {uploading && <span style={{ marginLeft: '1rem' }}>Processing…</span>}
        {message && (
          <p style={{ color: message.startsWith('Error') ? 'red' : 'green' }}>{message}</p>
        )}
      </div>

      <h2>Extracted Records</h2>
      {records.length === 0 ? (
        <p>No records yet.</p>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#f0f0f0' }}>
              <th style={th}>ID</th>
              <th style={th}>Filename</th>
              <th style={th}>Status</th>
              <th style={th}>Created At</th>
              <th style={th}>Data</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => (
              <tr key={r.id}>
                <td style={td}>{r.id}</td>
                <td style={td}>{r.filename}</td>
                <td style={td}>{r.status}</td>
                <td style={td}>{r.created_at}</td>
                <td style={td}>
                  <pre style={{ fontSize: '0.75rem', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                    {JSON.stringify(r.extracted_data, null, 2)}
                  </pre>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

const th = { border: '1px solid #ccc', padding: '0.5rem', textAlign: 'left' };
const td = { border: '1px solid #eee', padding: '0.5rem', verticalAlign: 'top' };

export default App;
