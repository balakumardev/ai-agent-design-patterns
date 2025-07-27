#!/usr/bin/env python3
"""
Setup script for Agent Design Patterns repository.
This script helps users set up their environment and test the installation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_step(step, text):
    """Print a formatted step."""
    print(f"\n[{step}] {text}")

def check_python_version():
    """Check if Python version is 3.9+."""
    print_step("1", "Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_virtual_environment():
    """Check if running in a virtual environment."""
    print_step("2", "Checking virtual environment...")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not in virtual environment. Consider creating one:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    return True

def install_dependencies():
    """Install required dependencies."""
    print_step("3", "Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_environment():
    """Set up environment file."""
    print_step("4", "Setting up environment configuration...")
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys before running patterns")
        return True
    else:
        print("❌ .env.example file not found")
        return False

def test_installation():
    """Test the installation by importing key modules."""
    print_step("5", "Testing installation...")
    
    test_modules = [
        "langchain",
        "langgraph", 
        "openai",
        "rich",
        "pydantic"
    ]
    
    failed_imports = []
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("\n✅ All core modules imported successfully")
    return True

def show_next_steps():
    """Show next steps to the user."""
    print_header("Setup Complete! Next Steps:")
    print("\n1. Edit the .env file with your API keys:")
    print("   - Add your OpenAI API key or configure LiteLLM")
    print("   - See .env.example for all available options")
    
    print("\n2. Test a pattern:")
    print("   cd 01-tool-using-agent")
    print("   python example.py")
    
    print("\n3. Run tests:")
    print("   cd 01-tool-using-agent")
    print("   python -m pytest test_tool_agent.py -v")
    
    print("\n4. Explore other patterns:")
    print("   - Each pattern has its own directory with README.md")
    print("   - Run example.py in each directory to see demonstrations")
    
    print("\n📚 Documentation:")
    print("   - Main README.md: Overview and quick start")
    print("   - Pattern READMEs: Detailed explanations and usage")
    
    print("\n🔧 Troubleshooting:")
    print("   - Check .env file configuration")
    print("   - Ensure API keys are valid")
    print("   - Check Python version (3.9+ required)")

def main():
    """Main setup function."""
    print_header("Agent Design Patterns Setup")
    print("This script will help you set up the environment for running AI agent patterns.")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    check_virtual_environment()
    
    # Install and configure
    if not install_dependencies():
        sys.exit(1)
    
    if not setup_environment():
        sys.exit(1)
    
    if not test_installation():
        print("\n⚠️  Some modules failed to import. You may need to:")
        print("   - Check your internet connection")
        print("   - Try: pip install --upgrade pip")
        print("   - Try: pip install -r requirements.txt --force-reinstall")
    
    show_next_steps()
    print(f"\n{'='*60}")
    print("🎉 Setup complete! Happy coding with AI agents!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
