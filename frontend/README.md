# Smart Livestock Gate web client

React and Vite client for training the current classifier and submitting farm
animal images to the Flask API.

## Development

```powershell
npm ci
npm run dev
```

The client calls `http://localhost:5000` by default. Copy `.env.example` to
`.env` and change `VITE_API_BASE_URL` when the API uses another address.

## Checks

```powershell
npm run lint
npm run build
```
