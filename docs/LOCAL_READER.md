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
   - The photo is first shrunk so its long side is at most 1,600 px, the size the models were trained on.
   - The largest envelope is cropped with an 8% margin.
2. **Text-line detector** (`line_det.onnx`, 5 MB): PaddleOCR PP-OCRv5 mobile detection. It finds every text line on the envelope.
3. **CAMID recognizer** (`camid_rec.onnx`, 8 MB).
   - PP-OCRv5 mobile recognition, fine-tuned on about 7,000 CAMID lines from Sanger/Ikiam envelopes, including crossed-out IDs.
   - It reads each line. Sideways lines are read turned both ways, and the more confident reading is kept.
   - A line is a CAMID reading only when the whole line is `CAM` + 6 digits.
4. **No envelope found:** the whole photo is read, upright and turned 180°.

## Rules before renaming (`src/lib/ocr/decide.ts`)

- **Database:** the most confident reading that exists in the specimen database. The CAMID list is read live from the database's Google Sheet, cached for a day, with a bundled copy used offline.
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

1. **Label crops automatically.** Collect line crops from renamed photos. Where the recognizer already reads the CAMID, the label is automatic. For the hard lines, the known CAMID is checked against the recognizer's character probabilities (CTC). A line holding only a crossed-out ID is labelled `#`, so the recognizer learns to skip it.
2. **Fine-tune.** Train from the current recognizer on a GPU. On the Sanger cluster this is about 20 minutes on an A100. The Paddle build used does not run on H100 GPUs.
3. **Export and check.** Export to ONNX with `paddle2onnx`, replace `public/models/camid_rec.onnx`, and run the parity test and unit tests.

The training and evaluation scripts are kept with the OCR project outside this repository (`sanger-envelope-sam3-20260924/ocr-next/`: `camid-train/`, `camid-bigtest/`, `fresh-test/`, `envelope-det-v2/`).
