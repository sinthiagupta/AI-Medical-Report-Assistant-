const { useState } = React;

const App = () => {
    const [labFile, setLabFile] = useState(null);
    const [xrayFile, setXrayFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState({ lab: null, xray: null });

    const API_BASE = "http://localhost:8000";

    const handleUpload = async (type, file) => {
        if (!file) return;
        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const endpoint = type === 'lab' ? '/analyze/lab' : '/analyze/xray';
            const response = await fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Upload failed");
            setResults(prev => ({ ...prev, [type]: data }));
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container">
            <header>
                <h1>AI Medical Assistant</h1>
                <p style={{color: '#8b949e'}}>Professional Diagnostic Support System</p>
            </header>

            <div className="grid">
                {/* Lab Reports Section */}
                <div className="card">
                    <h2>Blood Report Analysis</h2>
                    <p>Interpret complex laboratory data from PDF reports.</p>
                    
                    <div className="upload-box" onClick={() => document.getElementById('lab-input').click()}>
                        <input 
                            id="lab-input" 
                            type="file" 
                            accept=".pdf" 
                            hidden 
                            onChange={(e) => setLabFile(e.target.files[0])} 
                        />
                        <div style={{fontSize: '2rem'}}>📄</div>
                        <p>{labFile ? labFile.name : "Select Lab Report (PDF)"}</p>
                    </div>

                    <button 
                        disabled={!labFile || loading}
                        onClick={() => handleUpload('lab', labFile)}
                    >
                        {loading && <div className="loader"></div>}
                        Analyze Report
                    </button>

                    {results.lab && (
                        <div className="results">
                            <strong>Summary:</strong>
                            <p>{results.lab.summary}</p>
                            <details>
                                <summary style={{cursor: 'pointer', fontSize: '0.8rem'}}>Advanced Findings</summary>
                                <div className="details">{results.lab.details}</div>
                            </details>
                        </div>
                    )}
                </div>

                {/* X-Ray Section */}
                <div className="card">
                    <h2>X-Ray Diagnostic</h2>
                    <p>Computer vision for regional classification and detection.</p>
                    
                    <div className="upload-box" onClick={() => document.getElementById('xray-input').click()}>
                        <input 
                            id="xray-input" 
                            type="file" 
                            accept="image/*" 
                            hidden 
                            onChange={(e) => setXrayFile(e.target.files[0])} 
                        />
                        <div style={{fontSize: '2rem'}}>🩻</div>
                        <p>{xrayFile ? xrayFile.name : "Select X-Ray Image"}</p>
                    </div>

                    <button 
                        disabled={!xrayFile || loading}
                        onClick={() => handleUpload('xray', xrayFile)}
                    >
                        {loading && <div className="loader"></div>}
                        Scan Image
                    </button>

                    {results.xray && (
                        <div className="results">
                            <strong>Vision Analysis:</strong>
                            <p>{results.xray.summary}</p>
                            <div className="details">{results.xray.details}</div>
                        </div>
                    )}
                </div>
            </div>

            <footer style={{marginTop: '50px', textAlign: 'center', color: '#484f58', fontSize: '0.8rem'}}>
                &copy; 2026 AI Medical Systems. Professional Grade Assistant.
            </footer>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
