"""Turn an exported track JSONL into a compact, codec-free demo payload.

The frontend animates this JSON on a canvas to show persistent IDs and motion
trails without depending on a browser-playable video codec. Boxes are
normalized to 0..1 so the client can scale them to any canvas size.

Usage:
    python scripts/build_tracking_demo.py \
        artifacts/tracking/01_demo.jsonl artifacts/tracking/demo_tracks.json \
        --video data/raw/cattle_eye_view/videos/01.mp4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build(jsonl_path: Path, output_path: Path, width: int, height: int, fps: float,
          sequence: str) -> dict:
    frames: dict[int, list[dict]] = defaultdict(list)
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        x1, y1, x2, y2 = record["bbox_xyxy"]
        frames[record["frame_index"]].append(
            {
                "id": record["track_id"],
                "state": record["track_state"],
                "conf": record["confidence"],
                # Normalized [x, y, w, h] in 0..1 for resolution-independent draw.
                "box": [
                    round(x1 / width, 4),
                    round(y1 / height, 4),
                    round((x2 - x1) / width, 4),
                    round((y2 - y1) / height, 4),
                ],
            }
        )

    ordered = [
        {"frame_index": index, "tracks": frames[index]}
        for index in sorted(frames)
    ]
    payload = {
        "sequence": sequence,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": len(ordered),
        "unique_ids": sorted({t["id"] for f in ordered for t in f["tracks"]}),
        "frames": ordered,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--sequence", default="01.mp4")
    args = parser.parse_args()

    width, height, fps = args.width, args.height, args.fps
    if args.video is not None and args.video.is_file():
        import cv2

        capture = cv2.VideoCapture(str(args.video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        fps = source_fps if source_fps and source_fps > 0 else fps
        capture.release()

    payload = build(args.jsonl, args.output, width, height, fps, args.sequence)
    print(
        f"wrote {args.output}: {payload['frame_count']} frames, "
        f"{len(payload['unique_ids'])} ids, {width}x{height}@{fps}fps"
    )


if __name__ == "__main__":
    main()
