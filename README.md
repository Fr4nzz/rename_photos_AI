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

1. The program crops the top left portion of each image where the CAM ID is written
2. These cropped images are merged into a grid and sent to Gemini AI
3. The AI identifies CAM IDs, notes any crossed-out IDs, and flags images to skip
4. Results can be reviewed and corrected in the Check Output tab
5. Final filenames follow the pattern: CAMID + suffix (d/v for dorsal/ventral)

## Notes

- Currently optimized for butterfly wing photo collections with specific layout and labeling
- Expects JPG and CR2 files to be in the same folder
- Default API keys included may have usage limitations
- Partial processing results are saved, allowing you to continue later
- Future versions may include more customization options for different workflows
