# Local CAMID reader

The web app reads the CAMID on each specimen photo inside the browser. No API key is needed, nothing is uploaded, and it runs in about 1 s per photo on a laptop. This page describes the models, the rules that decide when a photo is renamed automatically, how accurate it is, and how to update it.

## Credits

- **Envelope segmentation:** a fine-tuned [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) model (AGPL-3.0).
- **Text-line detection and CAMID recognition:** fine-tuned [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv5 models (Apache-2.0).
- **Runtime:** [ONNX Runtime Web](https://onnxruntime.ai) (MIT).
- **HEIC decoding:** [libheif](https://github.com/strukturag/libheif) via libheif-js (LGPL-3.0).

## Pipeline

1. **Envelope detector** (`public/models/envelope_det.onnx`, 11 MB).
   - A YOLO segmentation model fine-tuned on SAM3 "paper label" masks, covering both the older photo setup and the newer one with a colour card.
   - Only its boxes are used, so the file is exported without the mask output (same boxes, about a quarter faster), and it looks at the photo at 512 px instead of 1,024 (the same right and wrong renames on the speed campaign's locked set, a quarter of the time). The app takes the input size from the model file.
   - The photo is first shrunk so its long side is at most 1,600 px, the size the models were trained on.
   - The largest envelope is cropped with an 8% margin.
2. **Text-line detector** (`line_det.onnx`, 5 MB): PaddleOCR PP-OCRv5 mobile detection. It finds every text line on the envelope.
3. **CAMID recognizer** (`camid_rec.onnx`, 8 MB).
   - PP-OCRv5 mobile recognition, fine-tuned on about 7,000 CAMID lines from Sanger/Ikiam envelopes, including crossed-out IDs.
   - It reads each line. Sideways lines are read turned both ways, and the more confident reading is kept.
   - A line is a CAMID reading only when the whole line is `CAM` + 6 digits.
   - Lines are read upright first, in the detector's order. The first one that reads a CAMID with confidence 0.99 or more ends the reading; the lines are read turned 180° only when none did (no decision changed on the speed campaign's locked set of 2,014 photos, 64% fewer recognizer runs).
4. **No envelope found:** the whole photo is read, upright and turned 180°.
5. **The envelope's whole text** (when asked for: `readPhoto(..., { text: true })`, the server reader's `--text`; `src/lib/ocr/envelopeText.ts`).
   - Every line of the envelope is read in one turn per envelope (0, 90, 180 or 270°), not line by line.
   - Lines are grouped by shape: wide, tall, and short ones (a lone ♂/♀, a number) that follow the group with more text. Each group's two turns are compared by the mean log-probability per character over all its lines.
   - A sure CAM ID line's turn settles its group; wide lines are read turned over only when they read poorly upright.
   - The CAM ID reading and its decisions are unchanged; the text reuses the crops it already read (about +55 ms a photo on one processor thread).
   - On 348 archive envelopes (dev sample, 24 Oct 2026): sex read as in the database 68 % → 78 % (wrong 15 % → 4 %), dates 88 % → 91 %, read with the full-text model.

## Where it runs (`src/lib/photoWorker.ts`, `src/lib/devicePlan.ts`)

- Each photo is decoded once; the same worker reads its CAMID and checks its wings.
- When the browser offers the computer's graphics card (WebGPU), the models run on it if that is faster. The first time on a computer, and again after a browser or card change, each model is timed on the card (a few seconds, while "Read photos" starts); the choice is kept in the browser. A weak card can stay unused, and a model the card cannot run moves to the processor.
- On the speed campaign's 250 test photos the card gives the same decisions as the processor; the models' numbers differ from the 5th decimal on.
- Otherwise the models run on the processor (WebAssembly), one worker per logical processor as memory allows: 2 when the browser reports less than 8 GB, at most 4, or 6 on processors with 12 threads. More single-thread workers read faster than fewer with several threads each, so the app needs no cross-origin isolation.

## Rules before renaming (`src/lib/ocr/decide.ts`)

- **Database:** the most confident reading that exists in the specimen database. The CAMID list is read live from the database's Google Sheet, cached for a day, with a bundled copy used offline.
- **Margin** (`src/lib/ocr/lexicon.ts`): the reading is taken only when the recognizer finds it at least 1 nat more probable (CTC) than every other database ID within two edits. Otherwise the photo goes to review as an unsure read, with that other ID suggested. On the CAMID campaign's locked set (October 2026) this cut wrong renames from 8 to 2 and false renames on photos without a CAMID from 13 to 5, for about 5 more photos per 1,000 to review.
- **Sequence:** photos are ordered by EXIF capture time. A CAMID goes to review when none of the 3 photos on either side has an ID within 15 of it, since a session covers nearby IDs.
- **Repeated ID:** a specimen's dorsal and ventral photos are taken back to back. The same CAMID on photos that aren't adjacent in capture order, or on more than two photos, sends those photos to review.
- **Review suggestions:**
  - the most CAMID-like reading, with handwriting lookalikes fixed (for example `O`→`0` and `G`→`6`);
  - database IDs within two edits of that reading;
  - unused database IDs between the neighbouring photos' IDs.

## Accuracy

The main test set has 651 photos from collections the models never saw in training: Panama 2025, Brazil 2025, Peru 2024, Guyana 2025 and insectary-reared specimens. The photos filed under the wrong CAMID were identified and excluded from the error count.

| | Renamed correctly | Wrong CAMID | To review |
|---|---|---|---|
| Browser reader with all rules | 94.8% | 0.31% | 4.9% |

- **Familiar collections:** on held-out photos from the Ikiam/Ecuador collections the recognizer trained on, 97% were correct and 0.3% wrong.
- **What causes the remaining errors:** single-digit misreads where the wrong ID belongs to a specimen from the same session, and digits written over other digits.
- **What fails silently:** IDs written with five or seven digits are never renamed automatically, because they don't form a whole-line CAMID reading.

## Updating the models

The models improve with each new collection, especially new handwriting. Photos that have already been renamed become training data automatically: the photo is filed under its CAMID, and the envelope confirms it.

0. **Cut line crops as the app reads them.** `npm run envelope-lines -- --out <dir> <photos or list.txt>` (in `web-app/`) writes each envelope's turn, its lines' readings (`lines.jsonl`) and each line's crop as read (PNG). `--rec <onnx> --chars <json>` reads with another recogniser; `--known <camids.json>` applies the database rule to the CAM ID.
1. **Label crops automatically.** Collect line crops from renamed photos. Where the recognizer already reads the CAMID, the label is automatic. For the hard lines, the known CAMID is checked against the recognizer's character probabilities (CTC). A line holding only a crossed-out ID is labelled `#`, so the recognizer learns to skip it.
2. **Fine-tune.** Train from the current recognizer on a GPU. On the Sanger cluster this is about 20 minutes on an A100. The Paddle build used does not run on H100 GPUs.
3. **Export and check.** Export to ONNX with `paddle2onnx`, replace `public/models/camid_rec.onnx`, and run the parity test and unit tests.

The training and evaluation scripts are kept with the OCR project outside this repository (`sanger-envelope-sam3-20260924/ocr-next/`: `camid-train/`, `camid-bigtest/`, `fresh-test/`, `envelope-det-v2/`).
