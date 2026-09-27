# AI Photo Processor Web App

Browser-based interface for selecting specimen photos, rotating JPEG/PNG files, sending image grids to Gemini, reviewing extracted labels, and renaming files safely.

Live app: https://fr4nzz.github.io/rename_photos_AI/

## Development

```bash
npm install
npm run dev
```

The app runs with Vite. During development it is served from `/`; production builds use `/rename_photos_AI/` as the base path for GitHub Pages.

## Build

```bash
npm run lint
npm run build
```

The static build is written to `dist/` and deployed by `.github/workflows/deploy-frontend.yml`.

## Tests

```bash
npm test
```

The unit tests cover:
- lossless orientation edits on real CR2/JPEG files (those tests are skipped when the samples are absent);
- rename planning;
- RAW preview extraction;
- the reader's decision rules.

## Local CAMID reader

The reader lives in `src/lib/ocr/`:

| File | Role |
|---|---|
| `pipeline.ts` | Envelope → lines → CAMID |
| `decide.ts` | Database, sequence and repeated-ID rules, plus review suggestions |
| `ocrWorker.ts` / `engine.ts` | Web workers running onnxruntime-web |
| `database.ts` | Live CAMID list from the specimen Google Sheet |

The models are in `public/models/`. See [../docs/LOCAL_READER.md](../docs/LOCAL_READER.md).

## Formats

JPEG and PNG are read directly. RAW files (CR2, NEF, ARW, DNG, PEF; ORF with a smaller preview) are read through their embedded JPEG preview and rotated through their orientation tag. HEIC is not decoded by Chrome, so convert HEIC to JPEG first.
