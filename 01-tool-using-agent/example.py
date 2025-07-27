#!/usr/bin/env python3
"""
Example usage of the Tool-Using Agent Pattern.

This script demonstrates how to use the ToolUsingAgent with various tools
including calculator, web search, and file operations.
"""

import os
from dotenv import load_dotenv
from tool_agent import ToolUsingAgent
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()


def main():
    """Run example demonstrations of the tool-using agent."""
    
    console.print(Panel.fit("Tool-Using Agent Example", style="bold blue"))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ Please set OPENAI_API_KEY in your .env file", style="bold red")
        console.print("Copy .env.example to .env and add your OpenAI API key")
        return
    
    # Initialize the agent
    console.print("Initializing agent...", style="yellow")
    agent = ToolUsingAgent()
    console.print("✅ Agent ready!", style="green")
    
    # Example queries that demonstrate different tools
    examples = [
        {
            "description": "Mathematical Calculation",
            "query": "Calculate the area of a circle with radius 5. Use π = 3.14159",
            "expected_tool": "calculator"
        },
        {
            "description": "File Operations",
            "query": "Create a file called 'example.txt' with the content 'This is a test file created by the agent.'",
            "expected_tool": "file_operations"
        },
        {
            "description": "Web Search",
            "query": "Search for information about artificial intelligence",
            "expected_tool": "web_search"
        },
        {
            "description": "Complex Multi-Tool Task",
            "query": "Calculate 15 * 23, then save the result to a file called 'calculation_result.txt'",
            "expected_tool": "multiple tools"
        }
    ]
    
    console.print("\n🚀 Running Examples", style="bold cyan")
    
    for i, example in enumerate(examples, 1):
        console.print(f"\n[bold]Example {i}: {example['description']}[/bold]")
        console.print(f"[dim]Expected tool: {example['expected_tool']}[/dim]")
        console.print(f"[cyan]Query:[/cyan] {example['query']}")
        console.print("-" * 60, style="dim")
        
        try:
            # Run the query
            response = agent.run(example['query'])
            console.print(f"[green]Response:[/green] {response}")
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
    
    # Clean up example files
    cleanup_files = ['example.txt', 'calculation_result.txt']
    console.print("\n🧹 Cleaning up example files...", style="yellow")
    
    for filename in cleanup_files:
        if os.path.exists(filename):
            os.remove(filename)
            console.print(f"Removed {filename}", style="dim")
    
    console.print("\n✨ Example completed!", style="bold green")
    console.print("\nTo run the interactive demo, use: python tool_agent.py")


if __name__ == "__main__":
    main()
