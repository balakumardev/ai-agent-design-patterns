"""Shared utilities for all agent design patterns."""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

console = Console()


def get_llm_config() -> dict:
    """Get LLM configuration from environment variables."""
    config = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "60"))
    }
    
    return config


def create_llm(model_name: Optional[str] = None, temperature: Optional[float] = None) -> ChatOpenAI:
    """Create a ChatOpenAI instance with proper configuration for LiteLLM or OpenAI."""
    config = get_llm_config()
    
    # Override defaults if provided
    if model_name:
        config["model"] = model_name
    if temperature is not None:
        config["temperature"] = temperature
    
    # Create ChatOpenAI instance
    llm_kwargs = {
        "model": config["model"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "timeout": config["timeout"]
    }
    
    # Add API key
    if config["api_key"]:
        llm_kwargs["api_key"] = config["api_key"]
    
    # Add base URL if specified (for LiteLLM or other OpenAI-compatible providers)
    if config["base_url"]:
        llm_kwargs["base_url"] = config["base_url"]
    
    try:
        llm = ChatOpenAI(**llm_kwargs)
        return llm
    except Exception as e:
        console.print(f"❌ Error creating LLM: {str(e)}", style="bold red")
        console.print(f"Config: {llm_kwargs}", style="dim")
        raise


def check_llm_connection() -> bool:
    """Check if LLM connection is working."""
    try:
        llm = create_llm()
        
        # Test with a simple query
        response = llm.invoke("Say 'Hello, World!' and nothing else.")
        
        if response and response.content:
            console.print("✅ LLM connection successful", style="green")
            console.print(f"Response: {response.content}", style="dim")
            return True
        else:
            console.print("❌ LLM connection failed: Empty response", style="red")
            return False
            
    except Exception as e:
        console.print(f"❌ LLM connection failed: {str(e)}", style="red")
        return False


def display_llm_config():
    """Display current LLM configuration."""
    config = get_llm_config()
    
    console.print("\n🔧 LLM Configuration:", style="bold blue")
    console.print(f"API Key: {'✅ Set' if config['api_key'] else '❌ Not set'}")
    console.print(f"Base URL: {config['base_url'] or 'Default (OpenAI)'}")
    console.print(f"Model: {config['model']}")
    console.print(f"Temperature: {config['temperature']}")
    console.print(f"Max Tokens: {config['max_tokens']}")
    console.print(f"Timeout: {config['timeout']}s")


def validate_environment() -> bool:
    """Validate that the environment is properly configured."""
    config = get_llm_config()
    
    if not config["api_key"]:
        console.print("❌ OPENAI_API_KEY not set in environment", style="bold red")
        console.print("Please set OPENAI_API_KEY in your .env file", style="yellow")
        return False
    
    # Test connection
    return check_llm_connection()


if __name__ == "__main__":
    """Test the LLM configuration."""
    console.print("🧪 Testing LLM Configuration", style="bold cyan")
    
    display_llm_config()
    
    if validate_environment():
        console.print("\n✅ Environment validation successful!", style="bold green")
    else:
        console.print("\n❌ Environment validation failed!", style="bold red")
