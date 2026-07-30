# Implementation roadmap

The application should evolve as one evaluated video pipeline, not as separate
model demos:

```text
video -> cattle detector -> persistent tracker -> virtual gate -> count + overlay
```

The React client consumes only HTTP APIs. Dataset paths, model paths, and raw
annotations remain backend concerns so the application works after cloning to a
different computer.

## Milestone 0: readiness dashboard (complete)

- `GET /api/health` confirms that the Flask service is available.
- `GET /api/status` reports path-free dataset, preparation, and model status.
- The frontend displays validated dataset totals and the next pipeline action.
- A clone without CattleEyeView still loads and explains that local installation
  is required.

## Milestone 1: reproducible cattle detector (implemented)

1. `train-detector` uses the prepared `detect/dataset.yaml`.
2. Weights, configuration, environment details, plots, and metrics are saved
   under `artifacts/`.
3. The best checkpoint is evaluated once on the official held-out test split
   with precision, recall, and mAP.
4. Deterministic example inference is exposed through allowlisted HTTP routes
   and visualized in the frontend alongside human annotations.

Acceptance criteria: a fresh authorized installation can prepare data, train a
detector, and reproduce the reported test metrics using documented commands.

## Milestone 2: tracking and identity evaluation (implemented)

1. Connect detector outputs to a multi-object tracker. Implemented as a
   SORT-style tracker (Kalman filter and Hungarian/IoU matching) in
   `src/smart_livestock_gate/tracking/`.
2. Normalize every output to the shared Step 2 schema: video, frame,
   timestamp, track ID, label, confidence, box, and lifecycle state. Implemented
   as `TrackedBox` in `tracking/schema.py`, exportable as JSONL or CSV.
3. Move each track through an explicit lifecycle
   (`tentative` -> `confirmed` -> `lost` -> `removed`) with tunable confidence,
   IoU, `min_hits`, and `max_age` parameters saved alongside results.
4. Evaluate persistent identities against the normalized CattleEyeView tracking
   manifest: identity switches, fragmentation, matched-track recall,
   mostly-tracked trajectories, and FPS. Implemented in `tracking/evaluate.py`
   and exposed via `livestock-gate track <sequence>`.
5. Render an annotated overlay MP4 (boxes, IDs, confidence, trails, frame,
   FPS) via `tracking/overlay.py` and `livestock-gate track-video`.
6. Expose results to the frontend: an animated, codec-free tracking demo and
   the identity metrics.

Acceptance criteria: each visible animal receives a stable ID across frames,
and tracking failures are measured rather than judged only by a demo video.
On four held-out test sequences (disjoint from the 09.mp4 clip used during
development), the default configuration reaches 78.6% matched-track recall,
recovers 179/221 ground-truth trajectories as mostly-tracked, and runs at
~21 FPS. Calm single-file sequences are near-perfect (05.mp4: 96.8% recall, 1
identity switch); the dense, occluded crowd in 01.mp4 is the main failure mode
(58.6% recall, 191 identity switches), most of it caused by missed detections
under top-down occlusion rather than the association step. The next tracker
improvement is therefore an appearance-based re-identification cost, not more
measurement plumbing.

## Milestone 3: virtual-gate counting

1. Represent the gate as two normalized endpoints so it works at any resolution.
2. Count only confirmed tracks whose center trajectory crosses the line in the
   configured direction.
3. Add hysteresis and a per-track counted flag to prevent double counting.
4. Compare per-video predictions against the 763 supplied ground-truth
   crossings using MAE, RMSE, and exact-count accuracy.

Acceptance criteria: every count links back to a track ID, frame, direction, and
confidence, with aggregate and per-video errors.

## Milestone 4: video jobs API

Add a local asynchronous job interface:

- `POST /api/jobs` accepts one supported video and gate configuration.
- `GET /api/jobs/<id>` reports queued, processing, complete, or failed.
- `GET /api/jobs/<id>/events` returns structured crossing events.
- `GET /api/jobs/<id>/video` streams the annotated result.

Uploaded files and generated results must have size limits, generated IDs,
controlled extensions, and automatic cleanup. Long inference work must not run
inside the request handler.

## Milestone 5: complete frontend workflow

1. Upload or select a sample video.
2. Draw the virtual gate and choose a direction.
3. Submit the job and display real progress by polling its status.
4. Play the annotated result and show inbound/outbound totals.
5. Present crossing events and downloadable evaluation JSON/CSV.

Acceptance criteria: a user can go from a video to a traceable count without
using the terminal, while the status page clearly explains any missing dataset
or model prerequisite.

## Milestone 6: class presentation

- Compare the detector against a simple baseline.
- Show detection, tracking, and counting metrics separately.
- Include representative successes and failures: occlusion, crowding, partial
  animals, lighting changes, and gate-line ambiguity.
- Demonstrate one held-out video live and explain where every displayed number
  came from.

This error decomposition is the strongest academic story: it shows whether a
wrong final count came from missed detection, broken identity tracking, or the
crossing rule.
