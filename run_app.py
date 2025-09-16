#!/usr/bin/env python3
"""
Simple script to run the Streamlit application.

Usage:
    python run_app.py

This script will start the Streamlit server for the Airbnb Market Intelligence Agent.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application."""
    
    # Check if we're in the right directory
    if not os.path.exists("src/app.py"):
        print("❌ Error: Please run this script from the project root directory")
        print("   The src/app.py file should be accessible from here.")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Please install requirements:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("🚀 Starting Aussie Airbnb Market Intelligence Agent...")
    print("📱 The app will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "="*60)
    
    # Run the Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running the app: {e}")

if __name__ == "__main__":
    main()
