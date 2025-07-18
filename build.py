# /build.py

import os
import sys
import subprocess
import platform

REQUIREMENTS_FILE = "requirements.txt"
SPEC_FILE = "build.spec" 
APP_NAME = "AI Photo Processor" # Updated App Name

def install_requirements():
    """Installs all packages from requirements.txt."""
    print(f"--- Installing packages from {REQUIREMENTS_FILE} ---")
    # Use check_call to ensure pip commands succeed
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
    print("--- Package installation complete ---\n")

def build_application():
    """Runs PyInstaller to build the application."""
    print(f"--- Building application with PyInstaller using {SPEC_FILE} ---")
    subprocess.check_call(['pyinstaller', SPEC_FILE])
    print("--- Build complete ---\n")

def main():
    """Main build process."""
    print(f"*** Starting build for {APP_NAME} on {platform.system()} ***\n")

    # 1. Install dependencies from requirements.txt
    install_requirements()

    # 2. Build the application using the spec file
    build_application()

    # 3. Final summary
    print("*********************************************************")
    print("🎉 Build process completed successfully! 🎉".center(55))
    print("*********************************************************")
    output_path = os.path.join('dist')
    print(f"\nYour application can be found in the '{os.path.abspath(output_path)}' directory.")


if __name__ == '__main__':
    main()