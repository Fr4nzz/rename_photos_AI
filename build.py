import os
import sys
import subprocess
import platform

def install_requirements():
    print("Installing required packages...")
    packages = [
        'pyinstaller',
        'pandas',
        'piexif',
        'pillow',
        'PyQt5',
        'google-generativeai'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def build_application():
    print("Building application...")
    os_type = platform.system().lower()
    
    # Run PyInstaller
    subprocess.check_call(['pyinstaller', 'rename_photos.spec'])
    
    # Create output directory structure
    if os_type == 'windows':
        dist_dir = os.path.join('dist', 'Rename_Photos_AI')
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)
    
    print("\nBuild completed!")
    if os_type == 'windows':
        print("Your executable is in: dist/Rename_Photos_AI.exe")
    else:
        print("Your application is in: dist/Rename_Photos_AI.app")

def main():
    try:
        # Install requirements
        install_requirements()
        
        # Build the application
        build_application()
        
        print("\nBuild process completed successfully!")
        
    except Exception as e:
        print(f"\nError during build process: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
