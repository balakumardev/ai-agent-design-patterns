#!/usr/bin/env python3
"""
Example usage of the Reflection Pattern.

This script demonstrates how to use the ReflectionAgent to improve
responses through iterative self-critique and revision.
"""

import os
from dotenv import load_dotenv
from reflection_agent import ReflectionAgent
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

# Load environment variables
load_dotenv()

console = Console()


def main():
    """Run example demonstrations of the reflection agent."""
    
    console.print(Panel.fit("Reflection Pattern Example", style="bold blue"))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ Please set OPENAI_API_KEY in your .env file", style="bold red")
        console.print("Copy .env.example to .env and add your OpenAI API key")
        return
    
    # Initialize the agent
    console.print("Initializing reflection agent...", style="yellow")
    agent = ReflectionAgent(max_iterations=3)
    console.print("✅ Agent ready!", style="green")
    
    # Example queries that benefit from reflection
    examples = [
        {
            "description": "Educational Explanation",
            "query": "Explain quantum computing to a high school student",
            "why_reflection_helps": "Complex topics benefit from iterative refinement to improve clarity and accessibility"
        },
        {
            "description": "Creative Writing",
            "query": "Write a short poem about the beauty of mathematics",
            "why_reflection_helps": "Creative content can be enhanced through multiple iterations of critique and improvement"
        },
        {
            "description": "Technical Comparison",
            "query": "Compare Python and JavaScript for web development",
            "why_reflection_helps": "Technical comparisons benefit from ensuring completeness and balanced perspective"
        },
        {
            "description": "Problem-Solving Strategy",
            "query": "How can someone overcome procrastination when studying?",
            "why_reflection_helps": "Advice can be improved by ensuring it's comprehensive and actionable"
        }
    ]
    
    console.print("\n🚀 Running Reflection Examples", style="bold cyan")
    console.print("Each example will show how the agent improves its response through self-reflection.\n")
    
    for i, example in enumerate(examples, 1):
        console.print(f"[bold]Example {i}: {example['description']}[/bold]")
        console.print(f"[dim]Why reflection helps: {example['why_reflection_helps']}[/dim]")
        console.print(f"[cyan]Query:[/cyan] {example['query']}")
        console.print("-" * 80, style="dim")
        
        try:
            # Run with reflection (max 2 iterations for demo)
            result = agent.run(example['query'], max_iterations=2)
            
            # Display results
            console.print(f"\n[green]✨ Final Response (after {result.iteration} iterations):[/green]")
            console.print(Panel(result.content, border_style="green", title="Improved Response"))
            
            if result.critique:
                console.print(f"\n[yellow]🔍 Final Critique:[/yellow]")
                console.print(Panel(result.critique, border_style="yellow", title="Self-Assessment"))
            
            # Show improvement indicator
            if result.iteration > 1:
                console.print(f"[bold green]✅ Response improved through {result.iteration} iterations[/bold green]")
            else:
                console.print(f"[bold blue]ℹ️  Response was good from the start (1 iteration)[/bold blue]")
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
        
        console.print("\n" + "="*80 + "\n")
    
    # Demonstrate streaming
    console.print("[bold magenta]🔄 Streaming Example[/bold magenta]")
    console.print("Watch the reflection process step by step:")
    console.print("-" * 80, style="dim")
    
    streaming_query = "Explain the benefits and drawbacks of artificial intelligence"
    console.print(f"[cyan]Streaming Query:[/cyan] {streaming_query}\n")
    
    try:
        agent.stream_reflection(streaming_query, max_iterations=2)
    except Exception as e:
        console.print(f"[red]Streaming Error:[/red] {str(e)}")
    
    console.print("\n✨ Examples completed!", style="bold green")
    console.print("\nKey benefits of the Reflection Pattern:")
    
    benefits = [
        "🎯 Improved response quality through self-critique",
        "🔄 Iterative refinement process",
        "🧠 Self-awareness and meta-cognition",
        "📈 Consistent quality improvement",
        "🎨 Better handling of creative and complex tasks"
    ]
    
    for benefit in benefits:
        console.print(f"  {benefit}")
    
    console.print("\nTo run the interactive demo, use: python reflection_agent.py")


if __name__ == "__main__":
    main()
