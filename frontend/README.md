# Smart Livestock Gate web client

React and Vite dashboard for the CattleEyeView detection, tracking, and counting
pipeline. It currently reports backend, dataset, annotation, preparation, and
model readiness through `/api/health` and `/api/status`.

## Development

```powershell
npm ci
npm run dev
```

The client calls `http://localhost:5000` by default. Copy `.env.example` to
`.env` and change `VITE_API_BASE_URL` when the API uses another address.

The complete video-upload and inference workflow will be added after detector,
tracking, and line-crossing services have stable tested interfaces. See
`docs/roadmap.md` in the repository root.

## Checks

```powershell
npm run lint
npm run build
```
