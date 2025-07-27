#!/usr/bin/env python3
"""
Example usage of the Hierarchical Planning Pattern.

This script demonstrates how to use the HierarchicalPlanner to break down
complex goals into manageable, hierarchical tasks with dependencies.
"""

import os
from dotenv import load_dotenv
from hierarchical_planner import HierarchicalPlanner
from task_models import PlanningRequest, TaskPriority
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load environment variables
load_dotenv()

console = Console()


def main():
    """Run example demonstrations of the hierarchical planner."""
    
    console.print(Panel.fit("Hierarchical Planning Pattern Example", style="bold blue"))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ Please set OPENAI_API_KEY in your .env file", style="bold red")
        console.print("Copy .env.example to .env and add your OpenAI API key")
        return
    
    # Initialize the planner
    console.print("Initializing hierarchical planner...", style="yellow")
    planner = HierarchicalPlanner()
    console.print("✅ Planner ready!", style="green")
    
    # Example goals that benefit from hierarchical planning
    examples = [
        {
            "title": "Software Project Launch",
            "goal": "Develop and launch a task management web application",
            "context": "Team of 3 developers, 8-week timeline, React frontend, Node.js backend, PostgreSQL database",
            "why_hierarchical": "Complex projects need structured breakdown into phases, dependencies, and parallel workstreams"
        },
        {
            "title": "Event Planning",
            "goal": "Organize a tech conference for 200 attendees",
            "context": "Budget: $50,000, Venue: Convention center, Duration: 2 days, 10 speakers, catering included",
            "why_hierarchical": "Events have many interdependent tasks that must be coordinated across different teams"
        },
        {
            "title": "Research Project",
            "goal": "Conduct a comprehensive study on AI ethics in healthcare",
            "context": "Academic research, 6-month timeline, literature review, surveys, interviews, publication target",
            "why_hierarchical": "Research projects benefit from systematic phases with clear dependencies and milestones"
        }
    ]
    
    console.print("\n🚀 Running Hierarchical Planning Examples", style="bold cyan")
    console.print("Each example shows how complex goals are broken down into manageable hierarchical tasks.\n")
    
    for i, example in enumerate(examples, 1):
        console.print(f"[bold]Example {i}: {example['title']}[/bold]")
        console.print(f"[dim]Why hierarchical planning helps: {example['why_hierarchical']}[/dim]")
        console.print(f"[cyan]Goal:[/cyan] {example['goal']}")
        console.print(f"[cyan]Context:[/cyan] {example['context']}")
        console.print("-" * 80, style="dim")
        
        try:
            # Create planning request
            request = PlanningRequest(
                goal=example['goal'],
                context=example['context'],
                max_depth=3,
                max_tasks_per_level=4
            )
            
            # Generate plan
            console.print("\n[yellow]🏗️ Generating hierarchical plan...[/yellow]")
            plan = planner.create_plan(request)
            
            # Display the plan hierarchy
            console.print(f"\n[green]📋 Generated Hierarchical Plan:[/green]")
            console.print(Panel(plan.get_task_hierarchy_display(), border_style="green", title="Task Hierarchy"))
            
            # Show plan statistics
            progress = plan.get_progress_summary()
            total_duration = sum(t.estimated_duration or 0 for t in plan.task_registry.values() if not t.has_subtasks())
            
            # Create statistics table
            stats_table = Table(title="Plan Statistics", show_header=True, header_style="bold magenta")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            stats_table.add_row("Total Tasks", str(progress['total_tasks']))
            stats_table.add_row("Root Tasks", str(len(plan.root_tasks)))
            stats_table.add_row("Estimated Duration", f"{total_duration} minutes ({total_duration/60:.1f} hours)")
            
            # Count tasks by priority
            priority_counts = {}
            for task in plan.task_registry.values():
                if not task.has_subtasks():  # Only count leaf tasks
                    priority = task.priority.value
                    priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            for priority, count in priority_counts.items():
                stats_table.add_row(f"{priority.title()} Priority", str(count))
            
            console.print(f"\n{stats_table}")
            
            # Show task dependencies
            dependent_tasks = [t for t in plan.task_registry.values() if t.dependencies and not t.has_subtasks()]
            if dependent_tasks:
                console.print(f"\n[yellow]🔗 Tasks with Dependencies:[/yellow]")
                for task in dependent_tasks[:3]:  # Show first 3
                    deps = ", ".join(task.dependencies)
                    console.print(f"  • {task.title} → depends on: {deps}")
                if len(dependent_tasks) > 3:
                    console.print(f"  ... and {len(dependent_tasks) - 3} more")
            
            # Demonstrate execution readiness
            ready_tasks = plan.get_ready_tasks()
            console.print(f"\n[blue]⚡ Tasks Ready for Immediate Execution:[/blue]")
            for task in ready_tasks[:3]:  # Show first 3
                priority_emoji = {"critical": "🔴", "high": "🟡", "medium": "🔵", "low": "⚪"}
                emoji = priority_emoji.get(task.priority.value, "❓")
                console.print(f"  {emoji} {task.title} ({task.priority.value} priority)")
            if len(ready_tasks) > 3:
                console.print(f"  ... and {len(ready_tasks) - 3} more")
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
        
        console.print("\n" + "="*80 + "\n")
    
    # Demonstrate plan execution
    console.print("[bold magenta]🔄 Execution Demonstration[/bold magenta]")
    console.print("Let's create and execute a simple plan to see the full workflow:")
    console.print("-" * 80, style="dim")
    
    simple_goal = "Write a technical blog post about machine learning"
    simple_context = "Target audience: developers, 1500 words, include code examples and diagrams"
    
    console.print(f"[cyan]Goal:[/cyan] {simple_goal}")
    console.print(f"[cyan]Context:[/cyan] {simple_context}\n")
    
    try:
        # Create and execute plan
        request = PlanningRequest(
            goal=simple_goal,
            context=simple_context,
            max_depth=2,
            max_tasks_per_level=3
        )
        
        console.print("[yellow]📋 Creating plan...[/yellow]")
        plan = planner.create_plan(request)
        
        console.print("\n[green]Initial Plan:[/green]")
        console.print(Panel(plan.get_task_hierarchy_display(), border_style="green"))
        
        console.print("\n[yellow]🚀 Executing plan...[/yellow]")
        executed_plan = planner.execute_plan(plan)
        
        console.print("\n[green]✅ Execution Results:[/green]")
        console.print(Panel(executed_plan.get_task_hierarchy_display(), border_style="green"))
        
        # Show final statistics
        final_progress = executed_plan.get_progress_summary()
        console.print(f"\n[bold blue]📊 Execution Summary:[/bold blue]")
        console.print(f"  Completed: {final_progress['completed_tasks']}/{final_progress['total_tasks']} tasks")
        console.print(f"  Success rate: {final_progress['progress_percentage']:.1f}%")
        console.print(f"  Failed tasks: {final_progress['failed_tasks']}")
        
    except Exception as e:
        console.print(f"[red]Execution Error:[/red] {str(e)}")
    
    console.print("\n✨ Examples completed!", style="bold green")
    console.print("\nKey benefits of Hierarchical Planning:")
    
    benefits = [
        "🏗️ Breaks complex goals into manageable tasks",
        "🔗 Manages dependencies between tasks",
        "📊 Provides clear progress tracking",
        "⚡ Identifies tasks ready for immediate execution",
        "🎯 Maintains focus on the overall goal",
        "📈 Enables parallel execution of independent tasks",
        "🔄 Supports iterative refinement of plans"
    ]
    
    for benefit in benefits:
        console.print(f"  {benefit}")
    
    console.print("\nTo run the interactive demo, use: python hierarchical_planner.py")


if __name__ == "__main__":
    main()
