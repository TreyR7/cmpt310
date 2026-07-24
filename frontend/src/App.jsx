// App.jsx
import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

function App() {
  const [trainStatus, setTrainStatus] = useState("idle"); // idle | training | trained | error
  const [selectedFile, setSelectedFile] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);

  const handleTrain = async () => {
    setTrainStatus("training");
    try {
      const res = await fetch(`${API_BASE}/api/train`, { method: "POST" });
      if (!res.ok) throw new Error("Training failed");
      await res.json();
      setTrainStatus("trained");
    } catch (err) {
      console.error(err);
      setTrainStatus("error");
    }
  };

  const handleUpload = (e) => {
    setSelectedFile(e.target.files[0]);
    setResult(null); // clear any stale result
  };

  const handleClassify = async () => {
    if (!selectedFile) return;
    setPredicting(true);
    try {
      const formData = new FormData();
      formData.append("images", selectedFile);

      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div>
      <button onClick={handleTrain} disabled={trainStatus === "training"}>
        {trainStatus === "training" ? "Training..." : "Train model"}
      </button>
      <p>Status: {trainStatus}</p>

      <input type="file" accept="image/*" onChange={handleUpload} />
      {selectedFile && <p>Selected: {selectedFile.name}</p>}

      <button onClick={handleClassify} disabled={!selectedFile || predicting}>
        {predicting ? "Classifying..." : "Classify"}
      </button>

      {result && (
        <div>
          <h4>Predictions</h4>
          <ul>
            {result.predictions?.map((p) => (
              <li key={p.filename}>{p.filename}: {p.label}</li>
            ))}
          </ul>
          <h4>Tally</h4>
          <pre>{JSON.stringify(result.tally, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
