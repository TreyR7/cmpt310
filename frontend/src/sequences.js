// CattleEyeView names each extracted-frame folder after its source video
// ("images/01.mp4/00902.jpg"), so the raw sequence key looks like a video file
// even though the UI is showing a single frame. Render a readable label instead.
export const sequenceLabel = (name) =>
  `Sequence ${String(name || "").replace(/\.mp4$/i, "")}`;
