import { useCallback, useEffect, useState } from "react";
import "./App.css";
import TrackingSection from "./Tracking.jsx";
import { sequenceLabel } from "./sequences.js";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";
const formatter = new Intl.NumberFormat();

const NEXT_STEPS = {
  install_dataset: {
    eyebrow: "Dataset required",
    title: "Install the authorized CattleEyeView release",
    detail: "Keep it under data/raw/cattle_eye_view, then validate the installation.",
  },
  validate_dataset: {
    eyebrow: "Validation required",
    title: "Validate the local dataset",
    detail: "Run livestock-gate dataset validate from the project environment.",
  },
  repair_dataset: {
    eyebrow: "Dataset issue",
    title: "Review the validation errors",
    detail: "Repair the missing or inconsistent files before training.",
  },
  prepare_detection_data: {
    eyebrow: "Prepare training data",
    title: "Create the YOLO detection layout",
    detail: "Run livestock-gate dataset prepare-yolo detect.",
  },
  train_detector: {
    eyebrow: "Next AI milestone",
    title: "Train and evaluate the cattle detector",
    detail: "The dataset is ready. The next implementation is a reproducible detector training command.",
  },
  build_tracking_pipeline: {
    eyebrow: "Next AI milestone",
    title: "Connect detection to tracking and counting",
    detail: "The detector is ready; persistent IDs and line-crossing inference come next.",
  },
  build_counting: {
    eyebrow: "Next AI milestone",
    title: "Add the virtual gate and counting",
    detail: "Detection and tracking are evaluated. Step 3 counts confirmed tracks that cross a directional line.",
  },
};

function Badge({ ready, children }) {
  return <span className={`badge ${ready ? "ready" : "pending"}`}>{children}</span>;
}

function Metric({ label, value }) {
  return (
    <article className="metric-card">
      <strong>{value == null ? "n/a" : formatter.format(value)}</strong>
      <span>{label}</span>
    </article>
  );
}

function DetectionExampleCard({ example, modelReady }) {
  const [prediction, setPrediction] = useState(null);
  const [showTruth, setShowTruth] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runDetection = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(
        `${API_BASE}/api/detection/examples/${example.id}/predict`,
        { method: "POST" },
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Detection failed");
      setPrediction(data);
      setShowTruth(false);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  const boxes = showTruth
    ? example.ground_truth.map((item) => ({ ...item, confidence: null }))
    : prediction?.detections || [];

  return (
    <article className="example-card">
      <div className="example-image">
        <img
          src={`${API_BASE}${example.image_url}`}
          alt={`CattleEyeView frame ${example.frame} from ${sequenceLabel(example.sequence)}`}
        />
        <div className="box-layer" aria-hidden="true">
          {boxes.map((detection, index) => {
            const [left, top, right, bottom] = detection.box;
            return (
              <span
                className={`detection-box ${showTruth ? "truth" : "prediction"}`}
                key={`${left}-${top}-${index}`}
                style={{
                  left: `${left * 100}%`,
                  top: `${top * 100}%`,
                  width: `${(right - left) * 100}%`,
                  height: `${(bottom - top) * 100}%`,
                }}
              >
                <span className="box-label">
                  {detection.label}
                  {detection.confidence != null && ` ${Math.round(detection.confidence * 100)}%`}
                </span>
              </span>
            );
          })}
        </div>
      </div>
      <div className="example-content">
        <div className="example-meta">
          <div>
            <strong>{sequenceLabel(example.sequence)}</strong>
            <span>frame {example.frame}</span>
          </div>
          {prediction && (
            <span className="inference-time">{prediction.inference_ms} ms</span>
          )}
        </div>
        <div className="example-counts">
          <span>Annotated: {example.ground_truth_count}</span>
          <span>Detected: {prediction?.count ?? "n/a"}</span>
        </div>
        {error && <p className="example-error">{error}</p>}
        <div className="example-actions">
          <button onClick={runDetection} disabled={!modelReady || loading}>
            {loading ? "Running AI…" : prediction ? "Run again" : "Run detector"}
          </button>
          <button
            className="secondary-button"
            onClick={() => setShowTruth((current) => !current)}
          >
            {showTruth ? "Show prediction" : "Show annotation"}
          </button>
        </div>
      </div>
    </article>
  );
}

function App() {
  const [dashboard, setDashboard] = useState({
    loading: true,
    error: "",
    health: null,
    status: null,
    examples: [],
  });

  const refresh = useCallback(async (signal) => {
    setDashboard((current) => ({ ...current, loading: true, error: "" }));
    try {
      const [healthResponse, statusResponse, examplesResponse] = await Promise.all([
        fetch(`${API_BASE}/api/health`, { signal }),
        fetch(`${API_BASE}/api/status`, { signal }),
        fetch(`${API_BASE}/api/detection/examples`, { signal }),
      ]);
      if (!healthResponse.ok || !statusResponse.ok || !examplesResponse.ok) {
        throw new Error("The backend returned an unexpected response.");
      }
      const [health, status, examplePayload] = await Promise.all([
        healthResponse.json(),
        statusResponse.json(),
        examplesResponse.json(),
      ]);
      setDashboard({
        loading: false,
        error: "",
        health,
        status,
        examples: examplePayload.examples,
      });
    } catch (error) {
      if (error.name !== "AbortError") {
        setDashboard({
          loading: false,
          error: `Cannot reach the API at ${API_BASE}. Start the Flask backend and try again.`,
          health: null,
          status: null,
          examples: [],
        });
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  const status = dashboard.status;
  const summary = status?.dataset.summary || {};
  const detector = status?.models.cattle_detector;
  const detectorMetrics = detector?.test_metrics || {};
  const action = NEXT_STEPS[status?.pipeline.next_step] || NEXT_STEPS.install_dataset;
  const pipeline = [
    ["Dataset", status?.pipeline.dataset_ready],
    ["Detection", status?.pipeline.detector_ready],
    ["Tracking", status?.pipeline.tracking_ready],
    ["Counting", status?.pipeline.counting_ready],
  ];

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Smart Livestock Gate home">
          <span className="brand-mark">SL</span>
          <span>Smart Livestock Gate</span>
        </a>
        <Badge ready={dashboard.health?.status === "ok"}>
          {dashboard.loading ? "Connecting" : dashboard.health ? "API online" : "API offline"}
        </Badge>
      </header>

      <main>
        <section className="hero-panel">
          <div>
            <p className="eyebrow">Cattle vision workspace</p>
            <h1>From video frames to reliable gate counts.</h1>
            <p className="hero-copy">
              One dashboard for dataset readiness, detector training, persistent tracking,
              and line-crossing evaluation.
            </p>
          </div>
          <button className="refresh-button" onClick={() => refresh()} disabled={dashboard.loading}>
            {dashboard.loading ? "Checking…" : "Refresh status"}
          </button>
        </section>

        {dashboard.error && <div className="error-banner" role="alert">{dashboard.error}</div>}

        {status && (
          <>
            <section className="section-block" aria-labelledby="dataset-heading">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Authorized local data</p>
                  <h2 id="dataset-heading">CattleEyeView dataset</h2>
                </div>
                <Badge ready={status.dataset.ready}>
                  {status.dataset.validation === "valid" ? "Validated" : status.dataset.validation.replaceAll("_", " ")}
                </Badge>
              </div>
              <div className="metrics-grid">
                <Metric label="video sequences" value={summary.sequences} />
                <Metric label="extracted frames" value={summary.frames} />
                <Metric label="tracked cattle" value={summary.unique_tracks} />
                <Metric label="ground-truth crossings" value={summary.ground_truth_crossings} />
              </div>
              {status.dataset.warnings.length > 0 && (
                <p className="dataset-note">Loader note: {status.dataset.warnings[0]}</p>
              )}
            </section>

            <div className="two-column">
              <section className="section-block" aria-labelledby="pipeline-heading">
                <div className="section-heading compact">
                  <div>
                    <p className="eyebrow">End-to-end progress</p>
                    <h2 id="pipeline-heading">Inference pipeline</h2>
                  </div>
                </div>
                <ol className="pipeline-list">
                  {pipeline.map(([label, ready], index) => (
                    <li key={label} className={ready ? "complete" : ""}>
                      <span className="step-number">{ready ? "✓" : index + 1}</span>
                      <span>{label}</span>
                      <small>{ready ? "Ready" : "Pending"}</small>
                    </li>
                  ))}
                </ol>
              </section>

              <section className="section-block next-action" aria-labelledby="action-heading">
                <p className="eyebrow">{action.eyebrow}</p>
                <h2 id="action-heading">{action.title}</h2>
                <p>{action.detail}</p>
              </section>
            </div>

            <section className="section-block" aria-labelledby="tasks-heading">
              <div className="section-heading compact">
                <div>
                  <p className="eyebrow">Annotation coverage</p>
                  <h2 id="tasks-heading">Available training tasks</h2>
                </div>
              </div>
              <div className="task-grid">
                {Object.entries(status.training_tasks).map(([name, task]) => (
                  <article className="task-card" key={name}>
                    <span>{name.replaceAll("_", " ")}</span>
                    <Badge ready={task.prepared}>{task.prepared ? "Prepared" : "Labels available"}</Badge>
                  </article>
                ))}
              </div>
            </section>

            <section className="section-block detection-lab" aria-labelledby="examples-heading">
              <div className="section-heading detection-heading">
                <div>
                  <p className="eyebrow">Held-out test frames</p>
                  <h2 id="examples-heading">See what the detector sees</h2>
                  <p className="section-copy">
                    Each example is one still frame extracted from a held-out gate video,
                    so the detector is scored on footage it never trained on. Green boxes
                    are AI predictions; amber boxes are the human annotations used for
                    comparison.
                  </p>
                </div>
                <Badge ready={detector?.ready}>
                  {detector?.ready ? `${detector.architecture} ready` : "Training required"}
                </Badge>
              </div>

              {detector?.ready && Object.keys(detectorMetrics).length > 0 && (
                <div className="model-metrics">
                  <Metric label="test precision" value={Math.round(detectorMetrics.precision * 1000) / 10} />
                  <Metric label="test recall" value={Math.round(detectorMetrics.recall * 1000) / 10} />
                  <Metric label="mAP at 50% IoU" value={Math.round(detectorMetrics.map50 * 1000) / 10} />
                  <Metric label="mAP 50-95" value={Math.round(detectorMetrics.map50_95 * 1000) / 10} />
                </div>
              )}

              <div className="example-grid">
                {dashboard.examples.map((example) => (
                  <DetectionExampleCard
                    example={example}
                    key={example.id}
                    modelReady={detector?.ready}
                  />
                ))}
              </div>
            </section>

            <TrackingSection />

            <section className="section-block" aria-labelledby="explanation-heading">
              <div className="section-heading compact">
                <div>
                  <p className="eyebrow">Inside one prediction</p>
                  <h2 id="explanation-heading">How the AI is working</h2>
                </div>
              </div>
              <div className="explanation-grid">
                <article>
                  <span>01</span>
                  <h3>Prepare the frame</h3>
                  <p>The image is resized with its aspect ratio preserved and converted into tensors.</p>
                </article>
                <article>
                  <span>02</span>
                  <h3>Extract visual features</h3>
                  <p>Learned convolutional layers respond to edges, textures, limbs, and cattle silhouettes.</p>
                </article>
                <article>
                  <span>03</span>
                  <h3>Propose cattle boxes</h3>
                  <p>The detection head predicts locations and a cattle confidence across multiple scales.</p>
                </article>
                <article>
                  <span>04</span>
                  <h3>Filter overlaps</h3>
                  <p>Low-confidence and duplicate boxes are removed before the final count is returned.</p>
                </article>
              </div>
            </section>
          </>
        )}
      </main>

      <footer>
        Local-first research prototype
        {dashboard.health && ` · API v${dashboard.health.version}`}
      </footer>
    </div>
  );
}

export default App;
