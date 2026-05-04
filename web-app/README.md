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

## Notes

- The hosted web app can process and rotate browser-supported image formats such as JPEG and PNG.
- RAW formats such as CR2 and ORF can be included in the workflow, but direct RAW rotation needs the optional local backend because browsers cannot safely rewrite those files by themselves.
- Gemini API keys are stored locally in the browser.
