import { useCallback, useEffect, useState } from "react";
import "./App.css";

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
};

function Badge({ ready, children }) {
  return <span className={`badge ${ready ? "ready" : "pending"}`}>{children}</span>;
}

function Metric({ label, value }) {
  return (
    <article className="metric-card">
      <strong>{value == null ? "—" : formatter.format(value)}</strong>
      <span>{label}</span>
    </article>
  );
}

function App() {
  const [dashboard, setDashboard] = useState({
    loading: true,
    error: "",
    health: null,
    status: null,
  });

  const refresh = useCallback(async (signal) => {
    setDashboard((current) => ({ ...current, loading: true, error: "" }));
    try {
      const [healthResponse, statusResponse] = await Promise.all([
        fetch(`${API_BASE}/api/health`, { signal }),
        fetch(`${API_BASE}/api/status`, { signal }),
      ]);
      if (!healthResponse.ok || !statusResponse.ok) {
        throw new Error("The backend returned an unexpected response.");
      }
      const [health, status] = await Promise.all([
        healthResponse.json(),
        statusResponse.json(),
      ]);
      setDashboard({ loading: false, error: "", health, status });
    } catch (error) {
      if (error.name !== "AbortError") {
        setDashboard({
          loading: false,
          error: `Cannot reach the API at ${API_BASE}. Start the Flask backend and try again.`,
          health: null,
          status: null,
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
