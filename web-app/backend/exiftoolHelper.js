const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

// Orientation constants (matches Python ANGLE_TO_ORIENTATION)
const ANGLE_TO_ORIENTATION = { 0: 1, 90: 8, 180: 3, 270: 6 };
const ORIENTATION_TO_ANGLE = { 1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 270, 7: 270, 8: 90 };

/**
 * Find exiftool binary: bundled (pkg snapshot) → app directory → system PATH.
 */
function getExiftoolPath() {
  const exeName = process.platform === 'win32' ? 'exiftool.exe' : 'exiftool';

  // 1. Check pkg snapshot path (when compiled with pkg)
  const snapshotPath = path.join(__dirname, 'bin', exeName);
  if (fs.existsSync(snapshotPath)) return snapshotPath;

  // 2. Check next to the executable
  const exeDir = path.dirname(process.execPath);
  const exeDirPath = path.join(exeDir, 'bin', exeName);
  if (fs.existsSync(exeDirPath)) return exeDirPath;

  // 3. Also check next to exe without /bin
  const exeDirDirect = path.join(exeDir, exeName);
  if (fs.existsSync(exeDirDirect)) return exeDirDirect;

  // 4. Check system PATH
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
