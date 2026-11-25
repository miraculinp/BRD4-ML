# BRD4 QSAR frontend (Vercel)

This is a standalone Next.js + TypeScript application. Push this folder's contents to the frontend Git repository and import that repository into Vercel.

## Vercel setup

1. Set the Vercel project Root Directory to `.` (the repository root is this frontend folder).
2. Set `NEXT_PUBLIC_API_URL` to the deployed backend's base URL, for example `https://brd4-qsar-api.onrender.com` (no trailing slash).
3. Deploy or redeploy after setting the environment variable.

## Local run

Requires Node.js 20.9 or newer.

```bash
npm install
npm run dev
```

The local frontend calls `http://localhost:8000` if `NEXT_PUBLIC_API_URL` is not set in development.

The interface accepts pasted SMILES and SDF/MOL2 uploads. It displays the three model predictions, applicability-domain results, and exports scored and not-scored rows to CSV. The upload limit is 20 MB and 100 molecules per request.
