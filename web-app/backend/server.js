const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const {
  getExiftoolPath,
  getOrientation,
  setOrientation,
  ANGLE_TO_ORIENTATION,
  ORIENTATION_TO_ANGLE,
} = require('./exiftoolHelper');

const app = express();
const DEFAULT_ALLOWED_ORIGINS = [
  'http://127.0.0.1:5173',
  'http://localhost:5173',
  'https://fr4nzz.github.io',
  'https://fr4nzz.github.io/rename_photos_AI',
];
const allowedOrigins = new Set([
  ...DEFAULT_ALLOWED_ORIGINS,
  ...(process.env.ALLOWED_ORIGINS || '')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean),
]);

app.use(cors({
  origin(origin, callback) {
    if (!origin || allowedOrigins.has(origin)) {
      callback(null, true);
      return;
    }
    callback(null, false);
  },
}));
app.use(express.json({ limit: '10mb' }));

const PORT = parseInt(process.env.PORT || '3847', 10);
const EXIFTOOL_PATH = getExiftoolPath();

const SUPPORTED_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.png', '.heic', '.heif',
  '.cr2', '.orf', '.tif', '.tiff', '.nef', '.arw', '.dng', '.raf',
]);

// ---------- Health ----------

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    version: '1.0.0',
    exiftool: !!EXIFTOOL_PATH,
    exiftoolPath: EXIFTOOL_PATH,
  });
});

// ---------- Rotate ----------

app.post('/rotate', async (req, res) => {
  if (!EXIFTOOL_PATH) {
    return res.status(500).json({ error: 'ExifTool not found' });
  }

  const { files, useExif = true } = req.body;
  if (!Array.isArray(files) || files.length === 0) {
    return res.status(400).json({ error: 'files array required' });
  }

  const results = [];
  for (const { path: filePath, angle } of files) {
    try {
      if (!fs.existsSync(filePath)) {
        results.push({ file: filePath, error: 'File not found' });
        continue;
      }

      const currentAngle = useExif
        ? ORIENTATION_TO_ANGLE[await getOrientation(EXIFTOOL_PATH, filePath)] ?? 0
        : 0;
      const finalAngle = (currentAngle + angle) % 360;
      const tag = ANGLE_TO_ORIENTATION[finalAngle] ?? 1;

      await setOrientation(EXIFTOOL_PATH, filePath, tag);
      results.push({ file: filePath, success: true, newTag: tag });
    } catch (err) {
      results.push({ file: filePath, error: err.message });
    }
  }

  res.json({ results });
});

// ---------- Rename ----------

app.post('/rename', (req, res) => {
  const { operations } = req.body;
  if (!Array.isArray(operations)) {
    return res.status(400).json({ error: 'operations array required' });
  }

  const results = [];
  const renameLog = [];

  for (const { from, to } of operations) {
    try {
      if (!fs.existsSync(from)) {
        results.push({ from, error: 'Source file not found' });
        continue;
      }

      const targetDir = path.dirname(from);
      const targetPath = path.join(targetDir, to);

      if (fs.existsSync(targetPath) && targetPath !== from) {
        results.push({ from, error: `Target already exists: ${to}` });
        continue;
      }

      fs.renameSync(from, targetPath);
      renameLog.push({ original: from, renamed: targetPath });
      results.push({ from, to: targetPath, success: true });
    } catch (err) {
      results.push({ from, error: err.message });
    }
  }

  // Save rename log for undo capability
  if (renameLog.length > 0) {
    const logDir = operations[0]?.from ? path.dirname(operations[0].from) : '.';
    const logPath = path.join(logDir, 'rename_files', `rename_log_${Date.now()}.json`);
    try {
      fs.mkdirSync(path.dirname(logPath), { recursive: true });
      fs.writeFileSync(logPath, JSON.stringify(renameLog, null, 2));
    } catch {}
  }

  res.json({ results, renamed: renameLog.length });
});

// ---------- List Files ----------

app.get('/files', (req, res) => {
  const { dir, type = 'all' } = req.query;
  if (!dir || !fs.existsSync(dir)) {
    return res.status(400).json({ error: 'Valid dir parameter required' });
  }

  try {
    const entries = fs.readdirSync(dir);
    const files = entries
      .filter((name) => {
        const ext = path.extname(name).toLowerCase();
        if (!SUPPORTED_EXTENSIONS.has(ext)) return false;
        if (type === 'compressed') return ['.jpg', '.jpeg', '.png', '.heic', '.heif'].includes(ext);
        if (type === 'raw') return ['.cr2', '.orf', '.tif', '.tiff', '.nef', '.arw', '.dng', '.raf'].includes(ext);
        return true;
      })
      .sort()
      .map((name) => ({
        name,
        path: path.join(dir, name),
      }));

    res.json({ files, count: files.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- Restore ----------

app.post('/restore', (req, res) => {
  const { logPath } = req.body;
  if (!logPath || !fs.existsSync(logPath)) {
    return res.status(400).json({ error: 'Valid logPath required' });
  }

  try {
    const log = JSON.parse(fs.readFileSync(logPath, 'utf-8'));
    const results = [];

    // Restore in reverse order
    for (const entry of [...log].reverse()) {
      try {
        if (fs.existsSync(entry.renamed)) {
          fs.renameSync(entry.renamed, entry.original);
          results.push({ from: entry.renamed, to: entry.original, success: true });
        } else {
          results.push({ from: entry.renamed, error: 'File not found' });
        }
      } catch (err) {
        results.push({ from: entry.renamed, error: err.message });
      }
    }

    res.json({ results, restored: results.filter((r) => r.success).length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- Start ----------

app.listen(PORT, '127.0.0.1', () => {
  console.log(`\n  AI Photo Processor Backend v1.0.0`);
  console.log(`  Running on http://127.0.0.1:${PORT}`);
  console.log(`  ExifTool: ${EXIFTOOL_PATH || 'NOT FOUND'}\n`);
});
