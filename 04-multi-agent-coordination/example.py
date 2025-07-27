#!/usr/bin/env python3
"""
Example usage of the Multi-Agent Coordination Pattern.

This script demonstrates how to use the MultiAgentCoordinator to orchestrate
multiple specialized agents working together to achieve complex goals.
"""

import os
from dotenv import load_dotenv
from multi_agent_coordinator import MultiAgentCoordinator
from agent_models import CoordinationRequest, AgentRole
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Load environment variables
load_dotenv()

console = Console()


def main():
    """Run example demonstrations of the multi-agent coordinator."""
    
    console.print(Panel.fit("Multi-Agent Coordination Pattern Example", style="bold blue"))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ Please set OPENAI_API_KEY in your .env file", style="bold red")
        console.print("Copy .env.example to .env and add your OpenAI API key")
        return
    
    # Initialize the coordinator
    console.print("Initializing multi-agent coordinator...", style="yellow")
    coordinator = MultiAgentCoordinator()
    console.print("✅ Coordinator ready!", style="green")
    
    # Example scenarios that benefit from multi-agent coordination
    examples = [
        {
            "title": "Product Launch Campaign",
            "goal": "Create and execute a comprehensive product launch campaign for a new AI tool",
            "context": "B2B SaaS product, target audience: developers and CTOs, budget: $100k, timeline: 6 weeks",
            "required_roles": ["researcher", "strategist", "writer", "analyst"],
            "why_coordination": "Requires diverse expertise working in parallel with shared information and dependencies"
        },
        {
            "title": "Technical Documentation Project",
            "goal": "Create comprehensive technical documentation for a complex API",
            "context": "REST API with 50+ endpoints, multiple authentication methods, SDK in 3 languages",
            "required_roles": ["technical_writer", "developer", "reviewer", "designer"],
            "why_coordination": "Multiple specialists need to collaborate while maintaining consistency and accuracy"
        },
        {
            "title": "Competitive Analysis Study",
            "goal": "Conduct a thorough competitive analysis of the AI chatbot market",
            "context": "15 major competitors, focus on features, pricing, market positioning, and user feedback",
            "required_roles": ["researcher", "analyst", "data_specialist", "strategist"],
            "why_coordination": "Different research streams need coordination to avoid duplication and ensure comprehensive coverage"
        }
    ]
    
    console.print("\n🚀 Running Multi-Agent Coordination Examples", style="bold cyan")
    console.print("Each example shows how specialized agents work together to achieve complex goals.\n")
    
    for i, example in enumerate(examples, 1):
        console.print(f"[bold]Example {i}: {example['title']}[/bold]")
        console.print(f"[dim]Why coordination helps: {example['why_coordination']}[/dim]")
        console.print(f"[cyan]Goal:[/cyan] {example['goal']}")
        console.print(f"[cyan]Context:[/cyan] {example['context']}")
        console.print(f"[cyan]Required roles:[/cyan] {', '.join(example['required_roles'])}")
        console.print("-" * 80, style="dim")
        
        try:
            # Create coordination request
            request = CoordinationRequest(
                goal=example['goal'],
                context=example['context'],
                required_roles=example['required_roles'],
                max_agents=4
            )
            
            # Execute coordination
            console.print("\n[yellow]🤝 Coordinating specialized agents...[/yellow]")
            plan = coordinator.coordinate(request)
            
            # Display the agent team
            console.print(f"\n[green]🤖 Assembled Agent Team:[/green]")
            agent_tree = Tree("Agent Team")
            
            for agent in plan.agents.values():
                agent_node = agent_tree.add(f"[bold cyan]{agent.name}[/bold cyan] ([green]{agent.role.value}[/green])")
                
                # Add capabilities
                caps_node = agent_node.add("[yellow]Capabilities:[/yellow]")
                for cap in agent.capabilities:
                    caps_node.add(f"• {cap.name}: {cap.description}")
                
                # Add status
                status_emoji = {
                    "idle": "💤", "busy": "⚡", "waiting": "⏳", 
                    "error": "❌", "offline": "📴"
                }
                emoji = status_emoji.get(agent.status.value, "❓")
                agent_node.add(f"[blue]Status:[/blue] {emoji} {agent.status.value.title()}")
            
            console.print(agent_tree)
            
            # Display coordination tasks
            console.print(f"\n[green]📋 Coordination Tasks & Results:[/green]")
            
            for j, task in enumerate(plan.tasks, 1):
                # Task header
                status_emoji = {
                    "pending": "⏳", "in_progress": "🔄", 
                    "completed": "✅", "failed": "❌"
                }
                emoji = status_emoji.get(task.status, "❓")
                
                console.print(f"\n[bold]Task {j}:[/bold] {emoji} {task.title}")
                console.print(f"[dim]Description: {task.description}[/dim]")
                
                # Assigned agent
                if task.assigned_agents:
                    agent_id = task.assigned_agents[0]
                    if agent_id in plan.agents:
                        agent_name = plan.agents[agent_id].name
                        agent_role = plan.agents[agent_id].role.value
                        console.print(f"[blue]Assigned to:[/blue] {agent_name} ({agent_role})")
                
                # Required capabilities
                if task.required_capabilities:
                    caps = ", ".join(task.required_capabilities)
                    console.print(f"[yellow]Required capabilities:[/yellow] {caps}")
                
                # Task result
                if task.result:
                    console.print(f"[green]Result:[/green]")
                    # Show result in a panel for better formatting
                    result_preview = task.result[:300] + "..." if len(task.result) > 300 else task.result
                    console.print(Panel(result_preview, border_style="green", title="Task Output"))
            
            # Show coordination statistics
            progress = plan.get_progress_summary()
            agent_status = plan.get_agent_status_summary()
            
            console.print(f"\n[bold blue]📊 Coordination Statistics:[/bold blue]")
            
            stats_table = Table(show_header=True, header_style="bold magenta")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            stats_table.add_row("Total Tasks", str(progress['total_tasks']))
            stats_table.add_row("Completed Tasks", str(progress['completed_tasks']))
            stats_table.add_row("Failed Tasks", str(progress['failed_tasks']))
            stats_table.add_row("Success Rate", f"{progress['progress_percentage']:.1f}%")
            stats_table.add_row("Total Agents", str(len(plan.agents)))
            stats_table.add_row("Messages Exchanged", str(len(plan.message_queue)))
            
            console.print(stats_table)
            
            # Show agent collaboration
            if plan.message_queue:
                console.print(f"\n[yellow]💬 Agent Collaboration (Recent Messages):[/yellow]")
                for msg in plan.message_queue[-3:]:  # Show last 3 messages
                    sender_name = "Unknown"
                    if msg.sender_id in plan.agents:
                        sender_name = plan.agents[msg.sender_id].name
                    
                    recipient = "All Agents" if msg.recipient_id == "ALL" else "Specific Agent"
                    if msg.recipient_id in plan.agents:
                        recipient = plan.agents[msg.recipient_id].name
                    
                    console.print(f"  📨 {sender_name} → {recipient}: {msg.content[:60]}...")
            
            # Show coordination benefits
            console.print(f"\n[bold green]✨ Coordination Benefits Demonstrated:[/bold green]")
            benefits = [
                f"🎯 Specialized expertise: {len(plan.agents)} agents with distinct roles",
                f"⚡ Parallel execution: {progress['total_tasks']} tasks coordinated",
                f"🔄 Information sharing: {len(plan.message_queue)} messages exchanged",
                f"📈 Efficiency: {progress['progress_percentage']:.1f}% success rate"
            ]
            
            for benefit in benefits:
                console.print(f"  {benefit}")
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
        
        console.print("\n" + "="*80 + "\n")
    
    # Demonstrate real-time coordination
    console.print("[bold magenta]🔄 Real-time Coordination Demonstration[/bold magenta]")
    console.print("Let's see how agents coordinate in real-time for a simple task:")
    console.print("-" * 80, style="dim")
    
    simple_goal = "Create a social media strategy for a tech conference"
    simple_context = "3-day conference, 500 attendees, focus on AI and machine learning"
    
    console.print(f"[cyan]Goal:[/cyan] {simple_goal}")
    console.print(f"[cyan]Context:[/cyan] {simple_context}\n")
    
    try:
        request = CoordinationRequest(
            goal=simple_goal,
            context=simple_context,
            required_roles=["researcher", "strategist", "writer"],
            max_agents=3
        )
        
        console.print("[yellow]🚀 Initiating coordination...[/yellow]")
        plan = coordinator.coordinate(request)
        
        console.print("\n[green]🤖 Agent Team Assembled:[/green]")
        for agent in plan.agents.values():
            console.print(f"  • {agent.name} ({agent.role.value}) - {len(agent.capabilities)} capabilities")
        
        console.print(f"\n[green]📋 Coordination Workflow:[/green]")
        for i, task in enumerate(plan.tasks, 1):
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌"}
            emoji = status_emoji.get(task.status, "❓")
            console.print(f"  {i}. {emoji} {task.title}")
            
            if task.result and task.status == "completed":
                console.print(f"     ✨ {task.result[:80]}...")
        
        final_progress = plan.get_progress_summary()
        console.print(f"\n[bold blue]📊 Final Results:[/bold blue]")
        console.print(f"  Coordination completed: {final_progress['is_complete']}")
        console.print(f"  Tasks completed: {final_progress['completed_tasks']}/{final_progress['total_tasks']}")
        console.print(f"  Agent collaboration messages: {len(plan.message_queue)}")
        
    except Exception as e:
        console.print(f"[red]Coordination Error:[/red] {str(e)}")
    
    console.print("\n✨ Examples completed!", style="bold green")
    console.print("\nKey benefits of Multi-Agent Coordination:")
    
    benefits = [
        "🤝 Specialized expertise working together",
        "⚡ Parallel task execution for efficiency",
        "💬 Information sharing and collaboration",
        "🎯 Role-based task assignment",
        "📊 Coordinated progress tracking",
        "🔄 Dynamic task redistribution",
        "🧠 Collective intelligence and problem-solving"
    ]
    
    for benefit in benefits:
        console.print(f"  {benefit}")
    
    console.print("\nTo run the interactive demo, use: python multi_agent_coordinator.py")


if __name__ == "__main__":
    main()
