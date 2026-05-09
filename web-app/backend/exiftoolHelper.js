const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Orientation constants (matches Python ANGLE_TO_ORIENTATION)
const ANGLE_TO_ORIENTATION = { 0: 1, 90: 8, 180: 3, 270: 6 };
const ORIENTATION_TO_ANGLE = { 1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 270, 7: 270, 8: 90 };

function copyRecursiveSync(from, to) {
  const stat = fs.statSync(from);
  if (stat.isDirectory()) {
    fs.mkdirSync(to, { recursive: true });
    for (const entry of fs.readdirSync(from)) {
      copyRecursiveSync(path.join(from, entry), path.join(to, entry));
    }
    return;
  }

  fs.mkdirSync(path.dirname(to), { recursive: true });
  fs.copyFileSync(from, to);
}

function materializeBundledExiftool(exeName) {
  const snapshotBinDir = path.join(__dirname, 'bin');
  const snapshotExiftool = path.join(snapshotBinDir, exeName);
  if (!process.pkg || !fs.existsSync(snapshotExiftool)) return null;

  const targetDir = path.join(os.tmpdir(), 'ai-photo-processor-backend', 'exiftool');
  const targetExiftool = path.join(targetDir, exeName);
  fs.mkdirSync(targetDir, { recursive: true });
  fs.copyFileSync(snapshotExiftool, targetExiftool);
  fs.chmodSync(targetExiftool, 0o755);

  const snapshotFilesDir = path.join(snapshotBinDir, 'exiftool_files');
  if (fs.existsSync(snapshotFilesDir)) {
    copyRecursiveSync(snapshotFilesDir, path.join(targetDir, 'exiftool_files'));
  }

  return targetExiftool;
}

/**
 * Find exiftool binary: bundled (pkg snapshot) → app directory → system PATH.
 */
function getExiftoolPath() {
  const exeName = process.platform === 'win32' ? 'exiftool.exe' : 'exiftool';

  // 1. Check pkg snapshot path. Snapshot assets cannot be executed directly,
  // so materialize the bundled binary and exiftool_files folder first.
  const bundledPath = materializeBundledExiftool(exeName);
  if (bundledPath) return bundledPath;

  // 2. Check development/local package path
  const snapshotPath = path.join(__dirname, 'bin', exeName);
  if (fs.existsSync(snapshotPath)) return snapshotPath;

  // 3. Check next to the executable
  const exeDir = path.dirname(process.execPath);
  const exeDirPath = path.join(exeDir, 'bin', exeName);
  if (fs.existsSync(exeDirPath)) return exeDirPath;

  // 4. Also check next to exe without /bin
  const exeDirDirect = path.join(exeDir, exeName);
  if (fs.existsSync(exeDirDirect)) return exeDirDirect;

  // 5. Check system PATH
  const { execSync } = require('child_process');
  try {
    const cmd = process.platform === 'win32' ? 'where exiftool' : 'which exiftool';
    const result = execSync(cmd, { encoding: 'utf-8' }).trim().split('\n')[0];
    if (result && fs.existsSync(result)) return result;
  } catch {}

  return null;
}

/**
 * Read EXIF orientation tag from a file.
 */
function getOrientation(exiftoolPath, filePath) {
  return new Promise((resolve, reject) => {
    execFile(exiftoolPath, ['-n', '-Orientation', filePath], { timeout: 10000 }, (err, stdout) => {
      if (err) return resolve(1); // default to 1 on error
      const match = stdout.match(/:\s*(\d+)/);
      resolve(match ? parseInt(match[1], 10) : 1);
    });
  });
}

/**
 * Write EXIF orientation tag to a file.
 */
function setOrientation(exiftoolPath, filePath, orientationTag) {
  return new Promise((resolve, reject) => {
    execFile(
      exiftoolPath,
      [`-Orientation=${orientationTag}`, '-overwrite_original', '-n', filePath],
      { timeout: 15000 },
      (err, stdout, stderr) => {
        if (err) return reject(new Error(`ExifTool failed: ${stderr || err.message}`));
        resolve(stdout);
      }
    );
  });
}

module.exports = {
  getExiftoolPath,
  getOrientation,
  setOrientation,
  ANGLE_TO_ORIENTATION,
  ORIENTATION_TO_ANGLE,
};
