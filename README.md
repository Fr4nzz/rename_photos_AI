# Wing Photos - CAM ID Processor

A user-friendly tool to automate the renaming of butterfly wing photos using Google's Gemini AI to extract CAM IDs from handwritten labels.

## Features

- **Batch Processing**: Processes multiple images at once by creating a grid of cropped photos
- **AI-Powered Recognition**: Uses Google's Gemini AI models to read handwritten CAM IDs from photos
- **Model Selection**: Choose between faster (Gemini Flash) or more accurate (Gemini Pro) models
- **Image Rotation**: Built-in tool to correctly rotate all JPG files at once
- **Manual Review**: Interface for reviewing and correcting AI-detected CAM IDs
- **Pair Verification**: Easily identify missing or duplicate wing pairs (dorsal/ventral)
- **Dual File Handling**: Renames both JPG and corresponding CR2 raw files
- **Restoration**: Option to restore original filenames if needed

## Download Standalone App
Ready-to-use standalone applications are available for both Windows and macOS:
[Download Standalone Apps](https://drive.google.com/drive/folders/1LnEkWZvFuysoqhRLQzDt3aqLw_tX2wW7?usp=sharing)
- Windows: `Rename_Photos_AI_Windows.zip`
- macOS: `Rename_Photos_AI_macOS.zip`
No installation required - just download, unzip, and run!

## How It Works

The app streamlines the process of identifying and renaming butterfly wing photos:

1. It focuses on the part of each image containing the handwritten CAM ID label
2. Multiple images are processed together for efficiency
3. Gemini AI reads the handwritten IDs and detects special cases (crossed-out IDs, etc.)
4. You can review all results before applying changes
5. Files are renamed following a consistent pattern for easier organization

## Notes

- Currently optimized for butterfly wing photo collections with specific layout and labeling
- Designed to work with both JPG and CR2 files stored in the same folder
- Includes API keys with usage limits
- Saves progress automatically, so you can pause and continue later
- Future versions may include more customization options for different workflows
