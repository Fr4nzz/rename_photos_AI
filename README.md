# AI Photo Processor

AI Photo Processor is a browser app for cataloging specimen photo batches. It helps select and rotate images, build preview grids, send images to Gemini for label extraction, review the results, and rename files with undo support.

Live app: https://fr4nzz.github.io/rename_photos_AI/

![AI Photo Processor Screenshot](screenshots/2025-07-17.png)

## What It Does

- Selects an image folder and lets you choose exactly which images are active for processing.
- Rotates selected JPEG/PNG images in the browser and keeps an undo log for reversible changes.
- Builds cropped image grids for Gemini OCR with configurable rows, columns, merged height, and messages in parallel.
- Reviews extracted fields in a paginated table with thumbnails, filters, sorting, autosave, and CSV export.
- Renames selected files safely, with optional companion RAW/sidecar renaming when files share the same basename.

## Hosted Web App

Open the app here:

https://fr4nzz.github.io/rename_photos_AI/

The hosted version works directly in a modern browser. Browser-supported formats such as JPEG and PNG can be previewed, processed, and rotated from the web app. RAW formats such as CR2 and ORF can be included in the naming workflow, but direct RAW rotation requires the optional local backend because browsers cannot rewrite RAW metadata safely on their own.

Gemini API keys are saved locally in your browser.

## Recommended Workflow

1. Open the app and add your Gemini API key in the API Keys tab.
2. Use Select & Rotate Images to open a folder, filter images by filename/type, choose the active selection, and rotate JPEG/PNG files if needed.
3. Use Process Images to configure crop, grid size, model, prompt, and parallel messages before sending batches to Gemini.
4. Use Review Results to inspect thumbnails, correct extracted fields, create or load CSV files, and export results.
5. Recalculate names, then rename files when the review looks correct.

## Current Defaults

The web app is tuned for small Gemini messages that work well with Gemini 3.1 Flash Lite:

| Setting | Default |
| --- | --- |
| Model | `gemini-3.1-flash-lite-preview` |
| Grids per message | `1` |
| Grid rows | `2` |
| Grid columns | `2` |
| Merged image height | `1600` |
| Parallel messages | `5` |
| Main column | `CAM` |

The app includes rate-limit pacing so parallel requests do not exceed the configured model family limit.

## Run Locally

```bash
git clone https://github.com/Fr4nzz/rename_photos_AI.git
cd rename_photos_AI/web-app
npm install
npm run dev
```

## Verify Before Deploying

```bash
cd web-app
npm run lint
npm run build
```

GitHub Pages deployment is handled by `.github/workflows/deploy-frontend.yml` when changes are pushed to `main`.

## Legacy Desktop App

Older Python desktop builds are still available from GitHub Releases:

https://github.com/Fr4nzz/rename_photos_AI/releases

Those builds bundled ExifTool for local file metadata operations. The current hosted web app is the preferred version for ordinary browser-based processing, while RAW rotation remains a local-backend/desktop capability.

## License Notes

This project uses:

- ExifTool by Phil Harvey for local metadata workflows.
- Google Gemini APIs for AI image extraction.
