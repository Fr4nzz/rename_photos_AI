# Selection and Rotation Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class image selection workflow so rotation, AI processing, and review/rename can operate on selected images instead of always operating on the whole folder.

**Architecture:** Move folder image state and selected filenames into the shared processing store, add selection utility functions with focused tests, create a new Select Images tab for choosing images and rotating browser-supported files, then make Process Images and Review Results consume selected image subsets. Rotation undo for browser-side JPEG/PNG uses backup copies in a `rotation_backups/` folder plus a persisted rotation log.

**Tech Stack:** React, Zustand, File System Access API, TypeScript, Vite, ESLint.

---

### Task 1: Selection Utilities and Store State

**Files:**
- Create: `web-app/src/lib/selection.ts`
- Modify: `web-app/src/stores/processingStore.ts`
- Modify: `web-app/src/types/index.ts`

- [ ] Add `SelectionMode`, `SelectionOp`, and `RotationLogEntry` types.
- [ ] Add pure functions for select all, clear, invert, matching text replacement, matching text addition, extension filtering, and selected-file derivation.
- [ ] Add shared store state: `imageFiles`, `selectedImageNames`, `rotationLog`, and actions to update them.
- [ ] Verify with `npm run lint` and `npm run build`.

### Task 2: Select Images Tab

**Files:**
- Create: `web-app/src/components/select/SelectImagesTab.tsx`
- Create: `web-app/src/components/select/ImageSelectionGrid.tsx`
- Create: `web-app/src/components/select/ImageSelectionToolbar.tsx`
- Modify: `web-app/src/App.tsx`

- [ ] Add a new tab before Process Images named `Select Images`.
- [ ] Move folder opening and file-list display into Select Images.
- [ ] Default new folder selections to all loaded images selected.
- [ ] Add controls: all/none toggle, invert, filename text input, replace selection, add matches, sort by name/date/type, filter by extension, and selected/all counts.
- [ ] Show compact thumbnails and filename/type/date metadata.
- [ ] Verify with `npm run lint` and `npm run build`.

### Task 3: Browser Rotation With Undo

**Files:**
- Modify: `web-app/src/components/select/SelectImagesTab.tsx`
- Modify: `web-app/src/hooks/useProcessTab.ts`
- Modify: `web-app/src/lib/imageProcessing.ts`
- Modify: `web-app/src/lib/csvHandler.ts`

- [ ] Rotate selected JPEG/PNG files in browser using canvas and File System Access API.
- [ ] Before overwriting each file, save the original into `rotation_backups/`.
- [ ] Persist a rotation log with original filename, backup filename, angle, timestamp, and status.
- [ ] Add undo rotation for the latest/all browser rotation entries by restoring backup copies.
- [ ] Skip RAW/HEIC in browser with clear toast messaging.
- [ ] Verify with `npm run lint` and `npm run build`.

### Task 4: Process and Review Respect Selection

**Files:**
- Modify: `web-app/src/hooks/useProcessTab.ts`
- Modify: `web-app/src/components/process/ProcessTab.tsx`
- Modify: `web-app/src/hooks/useReviewTab.ts`
- Modify: `web-app/src/components/review/ReviewToolbar.tsx`
- Modify: `web-app/src/components/review/ReviewActionBar.tsx`

- [ ] Make AI processing operate on selected images by default.
- [ ] Add a Process toggle for selected/all if needed by state.
- [ ] Add Review toggle for showing all rows or selected rows.
- [ ] Make rename action respect selected rows when the review toggle is selected-only.
- [ ] Add a clear checkbox for companion renaming: rename files with the same basename and other supported extensions.
- [ ] Verify with `npm run lint` and `npm run build`.

### Task 5: Final Verification and Commit

**Files:**
- All changed files.

- [ ] Run `npm run lint`.
- [ ] Run `npm run build`.
- [ ] Check `git status --short --branch`.
- [ ] Commit with `feat: add selected image workflow`.
