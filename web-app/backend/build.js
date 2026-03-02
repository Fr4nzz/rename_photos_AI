const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');

const BIN_DIR = path.join(__dirname, 'bin');
const DIST_DIR = path.join(__dirname, 'dist');

// ExifTool download URL for Windows
const EXIFTOOL_VERSION = '13.10';
const EXIFTOOL_URL = `https://exiftool.org/exiftool-${EXIFTOOL_VERSION}.zip`;

async function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        return downloadFile(res.headers.location, dest).then(resolve).catch(reject);
      }
      res.pipe(file);
      file.on('finish', () => { file.close(resolve); });
    }).on('error', reject);
  });
}

async function main() {
  // Ensure directories exist
  fs.mkdirSync(BIN_DIR, { recursive: true });
  fs.mkdirSync(DIST_DIR, { recursive: true });

  // Check if exiftool already exists in bin/
  const exiftoolPath = path.join(BIN_DIR, 'exiftool.exe');
  if (!fs.existsSync(exiftoolPath)) {
    console.log(`ExifTool not found in bin/. Please download exiftool.exe manually.`);
    console.log(`Place exiftool.exe in: ${BIN_DIR}`);
    console.log(`Download from: https://exiftool.org`);
    console.log('');
    console.log('For automated builds, the GitHub Actions workflow handles this.');
    console.log('Continuing build without bundled exiftool...');
  } else {
    console.log(`Found exiftool at: ${exiftoolPath}`);
  }

  // Build with pkg
  console.log('\nBuilding executable with pkg...');
  try {
    execSync('npx pkg . --targets node18-win-x64 --output dist/AIPhotoProcessor-Backend.exe', {
      cwd: __dirname,
      stdio: 'inherit',
    });
    console.log('\nBuild complete! Output: dist/AIPhotoProcessor-Backend.exe');

    // Copy bin/ to dist/bin/ so exiftool is next to the exe
    if (fs.existsSync(exiftoolPath)) {
      const distBin = path.join(DIST_DIR, 'bin');
      fs.mkdirSync(distBin, { recursive: true });
      fs.copyFileSync(exiftoolPath, path.join(distBin, 'exiftool.exe'));
      console.log('Copied exiftool.exe to dist/bin/');
    }
  } catch (err) {
    console.error('Build failed:', err.message);
    process.exit(1);
  }
}

main();
