#!/usr/bin/env python3
"""
🎯 ULTIMATE JAPANESE QUIZ SOLVER - SETUP SCRIPT 🎯
Automatic installation and configuration for the most advanced Japanese quiz solver.

This script will:
- Install all required Python packages
- Download and install Tesseract OCR with Japanese support
- Configure API keys and environment variables
- Set up the database and directories
- Verify all components are working
- Launch the application

Author: Ultimate Quiz Solver Team
Version: 2.0 - The Ultimate Edition
"""

import os
import sys
import subprocess
import json
import sqlite3
import webbrowser
import zipfile
import urllib.request
from pathlib import Path
import logging

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color=Colors.END):
    """Print colored messages to terminal"""
    print(f"{color}{message}{Colors.END}")

def print_header():
    """Print the application header"""
    header = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        🎯 ULTIMATE JAPANESE QUIZ SOLVER SETUP 🎯              ║
    ║                                                               ║
    ║     The Most Advanced Japanese Question Solving System        ║
    ║                                                               ║
    ║  ✅ Full screen scanning with auto detection                   ║
    ║  ✅ Multi-AI provider support (Gemini, OpenAI, Claude)        ║
    ║  ✅ Advanced OCR with Japanese language support               ║
    ║  ✅ Question type detection and confidence scoring            ║
    ║  ✅ Professional GUI with history and analytics              ║
    ║  ✅ Global hotkeys and smart caching                          ║
    ║                                                               ║
    ║              Version 2.0 - Ultimate Edition                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print_colored(header, Colors.BLUE + Colors.BOLD)

def check_python_version():
    """Check if Python version is compatible"""
    print_colored("\n🐍 Checking Python version...", Colors.YELLOW)
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}.{version.micro}", Colors.RED)
        print_colored("Please install Python 3.8 or newer from https://python.org", Colors.YELLOW)
        return False
    
    print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible", Colors.GREEN)
    return True

def install_python_packages():
    """Install all required Python packages"""
    print_colored("\n📦 Installing Python packages...", Colors.YELLOW)
    
    packages = [
        "mss>=9.0.1",
        "pytesseract>=0.3.10",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "google-generativeai>=0.3.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "keyboard>=0.13.5",
        "langdetect>=1.0.9",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "sqlite3" if sys.version_info < (3, 7) else None,
    ]
    
    # Filter out None values
    packages = [pkg for pkg in packages if pkg]
    
    for package in packages:
        try:
            print_colored(f"  Installing {package}...", Colors.BLUE)
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True, text=True)
            print_colored(f"  ✅ {package} installed successfully", Colors.GREEN)
        except subprocess.CalledProcessError as e:
            print_colored(f"  ⚠️ Failed to install {package}: {e}", Colors.YELLOW)
    
    print_colored("\n✅ Python packages installation completed!", Colors.GREEN)

def download_file(url, filename):
    """Download a file with progress indication"""
    print_colored(f"  Downloading {filename}...", Colors.BLUE)
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\r    Progress: {percent:.1f}%", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, filename, show_progress)
        print()  # New line after progress
        return True
    except Exception as e:
        print_colored(f"\n  ❌ Download failed: {e}", Colors.RED)
        return False

def install_tesseract():
    """Install Tesseract OCR with Japanese language support"""
    print_colored("\n🔍 Setting up Tesseract OCR...", Colors.YELLOW)
    
    # Check if Tesseract is already installed
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            print_colored(f"✅ Tesseract already installed at {path}", Colors.GREEN)
            
            # Check for Japanese language pack
            lang_dir = os.path.dirname(path) + r"\tessdata"
            if os.path.exists(os.path.join(lang_dir, "jpn.traineddata")):
                print_colored("✅ Japanese language pack found", Colors.GREEN)
                return True
            else:
                print_colored("⚠️ Japanese language pack missing", Colors.YELLOW)
                # Try to download Japanese pack
                return download_japanese_pack(lang_dir)
    
    # Tesseract not found - provide installation instructions
    print_colored("❌ Tesseract OCR not found", Colors.RED)
    print_colored("\n📋 TESSERACT INSTALLATION REQUIRED:", Colors.BOLD)
    print_colored("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki", Colors.YELLOW)
    print_colored("2. During installation, select 'Additional language data (download)'", Colors.YELLOW)
    print_colored("3. Make sure Japanese (jpn) is selected", Colors.YELLOW)
    print_colored("4. Install to default location (C:\\Program Files\\Tesseract-OCR)", Colors.YELLOW)
    
    # Ask user if they want to open the download page
    response = input("\nOpen Tesseract download page in browser? (y/n): ").lower().strip()
    if response == 'y':
        webbrowser.open("https://github.com/UB-Mannheim/tesseract/wiki")
    
    return False

def download_japanese_pack(tessdata_dir):
    """Download Japanese language pack for Tesseract"""
    print_colored("\n📥 Downloading Japanese language pack...", Colors.YELLOW)
    
    jpn_url = "https://github.com/tesseract-ocr/tessdata/raw/main/jpn.traineddata"
    jpn_path = os.path.join(tessdata_dir, "jpn.traineddata")
    
    try:
        if download_file(jpn_url, jpn_path):
            print_colored("✅ Japanese language pack downloaded successfully", Colors.GREEN)
            return True
        else:
            return False
    except Exception as e:
        print_colored(f"❌ Failed to download Japanese pack: {e}", Colors.RED)
        return False

def setup_api_keys():
    """Setup API keys for AI providers"""
    print_colored("\n🤖 Setting up AI providers...", Colors.YELLOW)
    
    api_keys = {}
    
    # Gemini API
    print_colored("\n🔹 GEMINI AI SETUP:", Colors.BLUE)
    print_colored("Get your free API key from: https://aistudio.google.com/app/apikey", Colors.YELLOW)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "YOUR_GEMINI_KEY_HERE":
        print_colored(f"✅ Gemini API key found in environment", Colors.GREEN)
        api_keys["gemini"] = gemini_key
    else:
        user_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
        if user_key:
            api_keys["gemini"] = user_key
            # Set environment variable
            os.system(f'setx GEMINI_API_KEY "{user_key}"')
            print_colored("✅ Gemini API key saved", Colors.GREEN)
    
    # OpenAI API
    print_colored("\n🔹 OPENAI SETUP (Optional):", Colors.BLUE)
    print_colored("Get API key from: https://platform.openai.com/api-keys", Colors.YELLOW)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print_colored("✅ OpenAI API key found in environment", Colors.GREEN)
        api_keys["openai"] = openai_key
    else:
        user_key = input("Enter your OpenAI API key (optional, press Enter to skip): ").strip()
        if user_key:
            api_keys["openai"] = user_key
            os.system(f'setx OPENAI_API_KEY "{user_key}"')
            print_colored("✅ OpenAI API key saved", Colors.GREEN)
    
    # Claude API
    print_colored("\n🔹 CLAUDE SETUP (Optional):", Colors.BLUE)
    print_colored("Get API key from: https://console.anthropic.com/", Colors.YELLOW)
    
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    if claude_key:
        print_colored("✅ Claude API key found in environment", Colors.GREEN)
        api_keys["claude"] = claude_key
    else:
        user_key = input("Enter your Claude API key (optional, press Enter to skip): ").strip()
        if user_key:
            api_keys["claude"] = user_key
            os.system(f'setx ANTHROPIC_API_KEY "{user_key}"')
            print_colored("✅ Claude API key saved", Colors.GREEN)
    
    if not api_keys:
        print_colored("⚠️ No API keys configured. You'll need at least one AI provider.", Colors.YELLOW)
        return False
    
    print_colored(f"\n✅ {len(api_keys)} AI provider(s) configured: {', '.join(api_keys.keys())}", Colors.GREEN)
    return True

def setup_directories():
    """Create necessary directories"""
    print_colored("\n📁 Creating directories...", Colors.YELLOW)
    
    directories = [
        "quiz_data",
        "quiz_data/logs",
        "quiz_data/exports",
        "quiz_data/cache",
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print_colored(f"  ✅ {directory}/", Colors.GREEN)
        except Exception as e:
            print_colored(f"  ❌ Failed to create {directory}: {e}", Colors.RED)

def setup_database():
    """Initialize the SQLite database"""
    print_colored("\n🗄️ Setting up database...", Colors.YELLOW)
    
    db_path = "quiz_data/quiz_history.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quiz_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    question_type TEXT,
                    confidence_score REAL,
                    ai_answer TEXT,
                    ai_provider TEXT,
                    processing_time REAL,
                    region_json TEXT,
                    ocr_confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON quiz_history(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_provider ON quiz_history(ai_provider)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type ON quiz_history(question_type)
            """)
            
        print_colored(f"✅ Database initialized at {db_path}", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"❌ Database setup failed: {e}", Colors.RED)
        return False

def create_launcher_scripts():
    """Create convenient launcher scripts"""
    print_colored("\n🚀 Creating launcher scripts...", Colors.YELLOW)
    
    # Console launcher
    console_launcher = """@echo off
title Ultimate Japanese Quiz Solver - Console
echo 🎯 Starting Ultimate Japanese Quiz Solver (Console Version)...
python ultimate_main.py
pause
"""
    
    # GUI launcher
    gui_launcher = """@echo off
title Ultimate Japanese Quiz Solver - GUI
echo 🎯 Starting Ultimate Japanese Quiz Solver (GUI Version)...
python ultimate_gui.py
pause
"""
    
    # Quick setup launcher
    setup_launcher = """@echo off
title Ultimate Japanese Quiz Solver - Setup
echo 🎯 Ultimate Japanese Quiz Solver Setup...
python ultimate_setup.py
pause
"""
    
    scripts = [
        ("🎯_Ultimate_Console.bat", console_launcher),
        ("🎯_Ultimate_GUI.bat", gui_launcher),
        ("🎯_Ultimate_Setup.bat", setup_launcher),
    ]
    
    for filename, content in scripts:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print_colored(f"  ✅ {filename}", Colors.GREEN)
        except Exception as e:
            print_colored(f"  ❌ Failed to create {filename}: {e}", Colors.RED)

def test_components():
    """Test all components are working correctly"""
    print_colored("\n🧪 Testing components...", Colors.YELLOW)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Import core modules
    try:
        import mss
        import pytesseract
        import cv2
        import numpy as np
        from PIL import Image
        print_colored("  ✅ Core modules import successfully", Colors.GREEN)
        success_count += 1
    except Exception as e:
        print_colored(f"  ❌ Core modules test failed: {e}", Colors.RED)
    
    # Test 2: Tesseract OCR
    try:
        version = pytesseract.get_tesseract_version()
        languages = pytesseract.get_languages()
        if 'jpn' in languages:
            print_colored(f"  ✅ Tesseract {version} with Japanese support", Colors.GREEN)
            success_count += 1
        else:
            print_colored(f"  ⚠️ Tesseract {version} found but no Japanese support", Colors.YELLOW)
    except Exception as e:
        print_colored(f"  ❌ Tesseract test failed: {e}", Colors.RED)
    
    # Test 3: AI providers
    try:
        available_providers = []
        
        # Test Gemini
        try:
            import google.generativeai as genai
            if os.getenv("GEMINI_API_KEY"):
                available_providers.append("Gemini")
        except:
            pass
        
        # Test OpenAI
        try:
            from openai import OpenAI
            if os.getenv("OPENAI_API_KEY"):
                available_providers.append("OpenAI")
        except:
            pass
        
        # Test Claude
        try:
            import anthropic
            if os.getenv("ANTHROPIC_API_KEY"):
                available_providers.append("Claude")
        except:
            pass
        
        if available_providers:
            print_colored(f"  ✅ AI providers available: {', '.join(available_providers)}", Colors.GREEN)
            success_count += 1
        else:
            print_colored("  ❌ No AI providers configured", Colors.RED)
    except Exception as e:
        print_colored(f"  ❌ AI providers test failed: {e}", Colors.RED)
    
    # Test 4: Screen capture
    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            if len(monitors) > 1:
                print_colored(f"  ✅ Screen capture ready ({len(monitors)-1} monitor(s))", Colors.GREEN)
                success_count += 1
            else:
                print_colored("  ❌ No monitors detected", Colors.RED)
    except Exception as e:
        print_colored(f"  ❌ Screen capture test failed: {e}", Colors.RED)
    
    # Test 5: Database
    try:
        import sqlite3
        db_path = "quiz_data/quiz_history.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("SELECT 1").fetchone()
            print_colored("  ✅ Database connection successful", Colors.GREEN)
            success_count += 1
    except Exception as e:
        print_colored(f"  ❌ Database test failed: {e}", Colors.RED)
    
    # Summary
    print_colored(f"\n📊 Component Tests: {success_count}/{total_tests} passed", Colors.BLUE)
    
    if success_count >= 4:
        print_colored("🎉 System is ready for Japanese quiz solving!", Colors.GREEN + Colors.BOLD)
        return True
    else:
        print_colored("⚠️ Some components need attention before using the system", Colors.YELLOW)
        return False

def show_completion_message():
    """Show setup completion message"""
    completion_message = f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        🎉 SETUP COMPLETED SUCCESSFULLY! 🎉                   ║
    ║                                                               ║
    ║     Ultimate Japanese Quiz Solver is now ready to use!       ║
    ║                                                               ║
    ║  📋 What you can do now:                                      ║
    ║                                                               ║
    ║  🎯 Run Ultimate GUI:     python ultimate_gui.py             ║
    ║  🎯 Run Console Version:  python ultimate_main.py            ║
    ║  🎯 Use Launcher Scripts: Double-click .bat files            ║
    ║                                                               ║
    ║  ⌨️ Global Hotkeys:                                          ║
    ║  • Ctrl+Shift+Q: Quick scan                                  ║
    ║  • Ctrl+Shift+R: Select region                               ║
    ║  • Ctrl+Shift+X: Emergency stop                              ║
    ║                                                               ║
    ║  📚 Features Available:                                       ║
    ║  • Full screen scanning with auto detection                  ║
    ║  • Multi-AI provider support                                 ║
    ║  • Advanced OCR preprocessing                                 ║
    ║  • Question type detection                                    ║
    ║  • Confidence scoring                                         ║
    ║  • History tracking and analytics                             ║
    ║  • Professional GUI interface                                 ║
    ║                                                               ║
    ║        Ready to solve Japanese questions perfectly! 🇯🇵        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print_colored(completion_message, Colors.GREEN + Colors.BOLD)

def main():
    """Main setup process"""
    print_header()
    
    # Setup steps
    steps = [
        ("Python Version Check", check_python_version),
        ("Python Packages", install_python_packages),
        ("Tesseract OCR", install_tesseract),
        ("AI API Keys", setup_api_keys),
        ("Directories", setup_directories),
        ("Database", setup_database),
        ("Launcher Scripts", create_launcher_scripts),
        ("Component Testing", test_components),
    ]
    
    print_colored(f"\n🔧 Starting setup process ({len(steps)} steps)...\n", Colors.BLUE + Colors.BOLD)
    
    completed_steps = 0
    
    for i, (step_name, step_function) in enumerate(steps, 1):
        print_colored(f"\n[{i}/{len(steps)}] {step_name}", Colors.BLUE + Colors.BOLD)
        print_colored("─" * 50, Colors.BLUE)
        
        try:
            if step_function():
                completed_steps += 1
                print_colored(f"✅ {step_name} completed successfully", Colors.GREEN)
            else:
                print_colored(f"⚠️ {step_name} completed with warnings", Colors.YELLOW)
        except Exception as e:
            print_colored(f"❌ {step_name} failed: {e}", Colors.RED)
    
    # Final summary
    print_colored(f"\n" + "="*70, Colors.BLUE)
    print_colored(f"SETUP SUMMARY: {completed_steps}/{len(steps)} steps completed successfully", Colors.BLUE + Colors.BOLD)
    
    if completed_steps >= len(steps) - 1:  # Allow 1 failure
        show_completion_message()
        
        # Ask to launch the application
        response = input("\nLaunch Ultimate Japanese Quiz Solver GUI now? (y/n): ").lower().strip()
        if response == 'y':
            try:
                print_colored("\n🚀 Launching Ultimate Japanese Quiz Solver GUI...", Colors.GREEN)
                subprocess.run([sys.executable, "ultimate_gui.py"], check=False)
            except Exception as e:
                print_colored(f"Failed to launch GUI: {e}", Colors.RED)
                print_colored("You can manually run: python ultimate_gui.py", Colors.YELLOW)
    else:
        print_colored("\n⚠️ Setup completed with some issues. Please review the errors above.", Colors.YELLOW)
        print_colored("You may still be able to use some features of the application.", Colors.YELLOW)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n\n⏹️ Setup cancelled by user", Colors.YELLOW)
    except Exception as e:
        print_colored(f"\n❌ Setup failed with error: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")
