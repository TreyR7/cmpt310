import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { sequenceLabel } from "./sequences.js";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

// Distinct colors keyed by track ID so one animal keeps one color; a sudden
// color change in the demo is exactly what an identity switch looks like.
const PALETTE = [
  "#4285f4", "#34a853", "#fbbc05", "#ea4335", "#a855f7",
  "#00bcd4", "#ff7043", "#9ccc65", "#ec407a", "#26c6da",
];
const colorFor = (id) => PALETTE[id % PALETTE.length];

// Plain-English descriptions of every metric, shown under each number.
const METRIC_HELP = {
  matched_track_recall: "Share of annotated cattle-frames the tracker recovered",
  id_switches: "Times one real animal's predicted ID changed (lower is better)",
  fragmentations: "Times one real path was split into separate tracks",
  mostly_tracked: "Ground-truth animals followed for ≥80% of their visible life",
  mean_processing_fps: "Frames per second for detection + tracking together",
};

const PIPELINE_STEPS = [
  {
    tag: "01",
    title: "Detect (YOLO)",
    body: "Each frame runs through the Step 1 cattle detector, producing boxes with a confidence score, but no memory of previous frames.",
  },
  {
    tag: "02",
    title: "Predict (Kalman filter)",
    body: "Every existing track carries a constant-velocity Kalman filter that predicts where its box should move next, so we can match even when the animal shifts between frames.",
  },
  {
    tag: "03",
    title: "Associate (IoU + Hungarian)",
    body: "We build an IoU overlap cost between predicted tracks and new detections, then the Hungarian algorithm picks the one-to-one matching with the lowest total cost.",
  },
  {
    tag: "04",
    title: "Lifecycle (birth / coast / death)",
    body: "Unmatched detections start tentative tracks; a track is confirmed after a few hits, kept alive through brief misses as lost, and removed once missing too long.",
  },
];

function AnimatedTracks({ demo }) {
  const canvasRef = useRef(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const rafRef = useRef(0);
  const lastTsRef = useRef(0);
  const accRef = useRef(0);

  const frameCount = demo.frames.length;
  const imagesRef = useRef(new Map());
  const [imagesLoaded, setImagesLoaded] = useState(0);

  // Fetch the real extracted video frames so boxes are drawn over the actual
  // cattle rather than an empty background. Frames are served pre-downscaled.
  useEffect(() => {
    let cancelled = false;
    const store = new Map();
    imagesRef.current = store;
    setImagesLoaded(0);
    let done = 0;

    demo.frames.forEach((entry, index) => {
      const image = new Image();
      // The API sends CORS headers, so request the frames as CORS-clean to keep
      // the canvas exportable (a tainted canvas blocks toDataURL/getImageData).
      image.crossOrigin = "anonymous";
      image.onload = () => {
        if (cancelled) return;
        store.set(index, image);
        done += 1;
        setImagesLoaded(done);
      };
      image.onerror = () => {
        if (cancelled) return;
        done += 1;
        setImagesLoaded(done);
      };
      image.src = `${API_BASE}/api/tracking/frames/${demo.sequence}/${entry.frame_index}?width=960`;
    });

    return () => {
      cancelled = true;
    };
  }, [demo]);

  // Precompute each ID's center at every frame so trails survive scrubbing.
  const centersById = useMemo(() => {
    const map = new Map();
    demo.frames.forEach((entry, index) => {
      for (const track of entry.tracks) {
        const [x, y, w, h] = track.box;
        if (!map.has(track.id)) map.set(track.id, new Map());
        map.get(track.id).set(index, [x + w / 2, y + h / 2]);
      }
    });
    return map;
  }, [demo]);

  const draw = useCallback(
    (frameIndex) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const { width, height } = canvas;
      ctx.fillStyle = "#101913";
      ctx.fillRect(0, 0, width, height);

      // The real video frame goes underneath; boxes and trails draw on top.
      const image = imagesRef.current.get(frameIndex);
      if (image) {
        ctx.drawImage(image, 0, 0, width, height);
      }

      const entry = demo.frames[frameIndex];
      if (!entry) return;

      let confirmed = 0;
      for (const track of entry.tracks) {
        const [nx, ny, nw, nh] = track.box;
        const x = nx * width;
        const y = ny * height;
        const w = nw * width;
        const h = nh * height;
        const color = colorFor(track.id);
        const isConfirmed = track.state === "confirmed";
        if (isConfirmed) confirmed += 1;

        // Trail: the last ~14 centers up to this frame.
        const history = centersById.get(track.id);
        if (history) {
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.5;
          ctx.lineWidth = 2;
          ctx.beginPath();
          let started = false;
          for (let f = Math.max(0, frameIndex - 14); f <= frameIndex; f += 1) {
            const c = history.get(f);
            if (!c) continue;
            const px = c[0] * width;
            const py = c[1] * height;
            if (!started) {
              ctx.moveTo(px, py);
              started = true;
            } else {
              ctx.lineTo(px, py);
            }
          }
          ctx.stroke();
          ctx.globalAlpha = 1;
        }

        ctx.strokeStyle = color;
        ctx.lineWidth = isConfirmed ? 2.5 : 1.5;
        if (!isConfirmed) ctx.setLineDash([6, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        const label = `#${track.id} · ${(track.conf * 100).toFixed(0)}%`;
        ctx.font = "600 13px system-ui, sans-serif";
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 18, textWidth + 10, 18);
        ctx.fillStyle = "#0c130e";
        ctx.fillText(label, x + 5, y - 5);
      }

      ctx.fillStyle = "rgba(12,19,14,0.85)";
      ctx.fillRect(0, 0, width, 26);
      ctx.fillStyle = "#ffffff";
      ctx.font = "600 13px system-ui, sans-serif";
      ctx.fillText(
        `${sequenceLabel(demo.sequence)}  ·  frame ${entry.frame_index}  ·  ${confirmed} confirmed tracks`,
        10,
        18,
      );
    },
    [demo, centersById],
  );

  useEffect(() => {
    // Repaint on frame change and whenever another preloaded frame arrives.
    void imagesLoaded;
    draw(frame);
  }, [frame, draw, imagesLoaded]);

  useEffect(() => {
    if (!playing) return undefined;
    const step = (ts) => {
      if (!lastTsRef.current) lastTsRef.current = ts;
      accRef.current += ts - lastTsRef.current;
      lastTsRef.current = ts;
      const interval = 1000 / (demo.fps || 8);
      if (accRef.current >= interval) {
        accRef.current = 0;
        setFrame((current) => (current + 1) % frameCount);
      }
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(rafRef.current);
      lastTsRef.current = 0;
    };
  }, [playing, demo.fps, frameCount]);

  const framesReady = imagesLoaded >= frameCount;

  return (
    <div className="tracker-demo">
      <div className="tracker-stage">
        <canvas ref={canvasRef} width={960} height={540} className="tracker-canvas" />
        {!framesReady && (
          <div className="tracker-loading">
            Loading video frames… {imagesLoaded}/{frameCount}
          </div>
        )}
      </div>
      <div className="tracker-controls">
        <button onClick={() => setPlaying((p) => !p)}>
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={frameCount - 1}
          value={frame}
          onChange={(event) => {
            setPlaying(false);
            setFrame(Number(event.target.value));
          }}
        />
        <span className="tracker-frame-count">
          {frame + 1}/{frameCount}
        </span>
      </div>
      <p className="tracker-legend">
        <span><i className="swatch solid" /> solid box = confirmed ID</span>
        <span><i className="swatch dashed" /> dashed = coasting through a miss</span>
        <span>tail = recent path · color/number = persistent identity</span>
      </p>
    </div>
  );
}

function MetricTile({ label, value, help, suffix }) {
  return (
    <article className="track-metric">
      <strong>
        {value == null ? "n/a" : value}
        {suffix}
      </strong>
      <span className="track-metric-label">{label}</span>
      {help && <span className="track-metric-help">{help}</span>}
    </article>
  );
}

export default function TrackingSection() {
  const [overview, setOverview] = useState(null);
  const [demo, setDemo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const overviewResponse = await fetch(`${API_BASE}/api/tracking/overview`, {
          signal: controller.signal,
        });
        const overviewData = await overviewResponse.json();
        setOverview(overviewData);
        if (overviewData.demo_available) {
          const demoResponse = await fetch(`${API_BASE}/api/tracking/demo`, {
            signal: controller.signal,
          });
          if (demoResponse.ok) setDemo(await demoResponse.json());
        }
      } catch (requestError) {
        if (requestError.name !== "AbortError") setError(requestError.message);
      }
    })();
    return () => controller.abort();
  }, []);

  const aggregate = overview?.aggregate;
  const config = overview?.config;
  const perSequence = overview?.per_sequence
    ? Object.entries(overview.per_sequence)
    : [];

  return (
    <section className="section-block tracking-block" aria-labelledby="tracking-heading">
      <div className="section-heading detection-heading">
        <div>
          <p className="eyebrow">Milestone 2 · persistent identity</p>
          <h2 id="tracking-heading">Tracking: keeping one ID per animal</h2>
          <p className="section-copy">
            The detector only answers "where are the cattle in this frame?" The
            tracker answers the harder question, "which detection is the same
            animal as last frame?", using a SORT-style pipeline of a Kalman
            filter and Hungarian matching.
          </p>
        </div>
        <span className={`badge ${overview?.ready ? "ready" : "pending"}`}>
          {overview?.ready ? "Evaluated" : "Not evaluated"}
        </span>
      </div>

      {error && <p className="example-error">{error}</p>}

      {demo ? (
        <AnimatedTracks demo={demo} />
      ) : (
        <p className="section-copy">
          Generate the demo with <code>livestock-gate track-video</code> to see the
          animated tracks here.
        </p>
      )}

      <div className="explanation-grid tracking-steps">
        {PIPELINE_STEPS.map((step) => (
          <article key={step.tag}>
            <span>{step.tag}</span>
            <h3>{step.title}</h3>
            <p>{step.body}</p>
          </article>
        ))}
      </div>

      {config && (
        <div className="tracker-config">
          <span className="tracker-config-title">Tuned parameters</span>
          <span className="config-chip">confidence ≥ {config.confidence}</span>
          <span className="config-chip">IoU ≥ {config.iou_threshold}</span>
          <span className="config-chip">confirm after {config.min_hits} hits</span>
          <span className="config-chip">drop after {config.max_age} misses</span>
        </div>
      )}

      {aggregate && (
        <>
          <div className="track-metrics-grid">
            <MetricTile
              label="Matched-track recall"
              value={(aggregate.matched_track_recall * 100).toFixed(1)}
              suffix="%"
              help={METRIC_HELP.matched_track_recall}
            />
            <MetricTile
              label="Identity switches"
              value={aggregate.id_switches}
              help={METRIC_HELP.id_switches}
            />
            <MetricTile
              label="Fragmentations"
              value={aggregate.fragmentations}
              help={METRIC_HELP.fragmentations}
            />
            <MetricTile
              label="Mostly tracked"
              value={`${aggregate.mostly_tracked}/${aggregate.ground_truth_tracks}`}
              help={METRIC_HELP.mostly_tracked}
            />
            <MetricTile
              label="Processing speed"
              value={aggregate.mean_processing_fps}
              suffix=" FPS"
              help={METRIC_HELP.mean_processing_fps}
            />
          </div>

          {perSequence.length > 0 && (
            <div className="track-table-wrap">
              <table className="track-table">
                <thead>
                  <tr>
                    <th>Sequence</th>
                    <th>GT tracks</th>
                    <th>Recall</th>
                    <th>ID sw.</th>
                    <th>Frag.</th>
                    <th>Mostly tracked</th>
                    <th>Mean IoU</th>
                  </tr>
                </thead>
                <tbody>
                  {perSequence.map(([name, seq]) => (
                    <tr key={name}>
                      <td>{sequenceLabel(name)}</td>
                      <td>{seq.ground_truth_tracks}</td>
                      <td>{(seq.matched_track_recall * 100).toFixed(1)}%</td>
                      <td>{seq.id_switches}</td>
                      <td>{seq.fragmentations}</td>
                      <td>
                        {seq.mostly_tracked}/{seq.ground_truth_tracks}
                      </td>
                      <td>{seq.mean_iou.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {overview?.note && <p className="dataset-note">{overview.note}</p>}
        </>
      )}

      {overview?.video_available && (
        <p className="dataset-note">
          The player above is the same tracker output drawn live in your browser.
          For the raw overlay,{" "}
          <a href={`${API_BASE}/api/tracking/video`}>
            download the annotated MP4
          </a>{" "}
          rendered by OpenCV at full 1920x1080. It uses the mp4v codec, so open it
          in a desktop player such as VLC or Windows Media Player; browsers cannot
          decode it, which is why the demo above is drawn on a canvas instead.
        </p>
      )}
    </section>
  );
}
