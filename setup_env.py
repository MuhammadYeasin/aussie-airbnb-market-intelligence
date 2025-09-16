#!/usr/bin/env python3
"""
Environment Setup Script for Aussie Airbnb Market Intelligence Agent

This script helps you create a .env file with your API keys.
"""

import os
import shutil

def main():
    """Set up the .env file for the project."""
    
    print("🏠 Aussie Airbnb Market Intelligence Agent - Environment Setup")
    print("=" * 60)
    
    # Check if .env already exists
    if os.path.exists(".env"):
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if response != 'y':
            print("❌ Setup cancelled.")
            return
    
    # Check if config template exists
    if not os.path.exists("config_template.txt"):
        print("❌ config_template.txt not found!")
        return
    
    print("\n📝 Setting up your .env file...")
    
    # Copy template to .env
    try:
        shutil.copy("config_template.txt", ".env")
        print("✅ Created .env file from template")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return
    
    # Get OpenAI API key from user
    print("\n🔑 OpenAI API Key Setup")
    print("You need an OpenAI API key to use this application.")
    print("Get your key from: https://platform.openai.com/api-keys")
    
    api_key = input("\nEnter your OpenAI API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Update the .env file with the actual API key
        try:
            with open(".env", "r") as f:
                content = f.read()
            
            # Replace the placeholder with the actual key
            content = content.replace("your_openai_api_key_here", api_key)
            
            with open(".env", "w") as f:
                f.write(content)
            
            print("✅ OpenAI API key saved to .env file")
            
        except Exception as e:
            print(f"❌ Error saving API key: {e}")
    else:
        print("⏭️  Skipped API key setup. You can edit .env file manually later.")
    
    print("\n🎉 Environment setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the app: python run_app.py")
    print("3. Or run Streamlit directly: streamlit run src/app.py")
    
    print("\n📁 Your .env file contains:")
    print("- OPENAI_API_KEY: Your OpenAI API key")
    print("- Optional settings for model configuration")
    print("- Placeholders for future API integrations")

if __name__ == "__main__":
    main()
