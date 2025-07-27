"""Hierarchical Planning Pattern Implementation using LangGraph."""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path for shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utils import create_llm, validate_environment

from task_models import Task, TaskStatus, TaskPriority, ExecutionPlan, PlanningRequest

load_dotenv()

# Initialize Rich console for better output
console = Console()


class PlanningState(TypedDict):
    """State of the hierarchical planning graph."""
    goal: str
    context: Optional[str]
    max_depth: int
    max_tasks_per_level: int
    current_depth: int
    execution_plan: ExecutionPlan
    current_task_id: Optional[str]
    planning_complete: bool


class HierarchicalPlanner:
    """A LangGraph-based agent that creates and executes hierarchical plans."""
    
    def __init__(self, model_name: str = None):
        """Initialize the hierarchical planner."""
        self.model = create_llm(model_name=model_name, temperature=0.3)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for hierarchical planning."""
        workflow = StateGraph(PlanningState)
        
        # Add nodes
        workflow.add_node("plan_decomposition", self.decompose_goal)
        workflow.add_node("task_execution", self.execute_task)
        workflow.add_node("progress_evaluation", self.evaluate_progress)
        
        # Set entry point
        workflow.set_entry_point("plan_decomposition")
        
        # Add edges
        workflow.add_edge("plan_decomposition", "task_execution")
        workflow.add_conditional_edges(
            "task_execution",
            self.should_continue_execution,
            {
                "continue": "progress_evaluation",
                "complete": END,
            },
        )
        workflow.add_conditional_edges(
            "progress_evaluation",
            self.should_continue_planning,
            {
                "execute_next": "task_execution",
                "replan": "plan_decomposition",
                "complete": END,
            },
        )
        
        return workflow.compile()
    
    def decompose_goal(self, state: PlanningState) -> Dict[str, Any]:
        """Decompose the goal into hierarchical tasks."""
        goal = state["goal"]
        context = state.get("context", "")
        max_depth = state["max_depth"]
        max_tasks = state["max_tasks_per_level"]
        current_depth = state["current_depth"]
        
        decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert task planner. Your job is to break down complex goals into manageable, hierarchical tasks.

Guidelines:
1. Create a logical hierarchy of tasks and subtasks
2. Each task should be specific and actionable
3. Identify dependencies between tasks
4. Assign appropriate priorities (critical, high, medium, low)
5. Estimate duration in minutes for leaf tasks
6. Maximum {max_tasks} tasks per level, maximum depth {max_depth}

IMPORTANT: Return ONLY a valid JSON object with this exact structure (no markdown, no explanations):
{{
    "tasks": [
        {{
            "title": "Task title",
            "description": "Detailed description",
            "priority": "high|medium|low|critical",
            "estimated_duration": 30,
            "dependencies": ["task_id1", "task_id2"],
            "subtasks": [
                {{
                    "title": "Subtask title",
                    "description": "Subtask description",
                    "priority": "medium",
                    "estimated_duration": 15,
                    "dependencies": []
                }}
            ]
        }}
    ]
}}"""),
            ("human", """Goal: {goal}
            
Context: {context}

Current planning depth: {current_depth}/{max_depth}

Please create a hierarchical task breakdown for this goal.""")
        ])
        
        try:
            response = self.model.invoke(
                decomposition_prompt.format_messages(
                    goal=goal,
                    context=context,
                    max_tasks=max_tasks,
                    max_depth=max_depth,
                    current_depth=current_depth
                )
            )
            
            # Parse the JSON response
            response_text = response.content.strip()

            # Try to extract JSON from the response if it's wrapped in markdown
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            try:
                task_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                console.print(f"❌ JSON parsing failed: {e}", style="red")
                console.print(f"Raw response: {response_text[:200]}...", style="dim")

                # Fallback: create a simple task structure
                task_data = {
                    "tasks": [
                        {
                            "id": "fallback_task_1",
                            "title": "Analyze the problem",
                            "description": "Break down the problem into smaller components",
                            "priority": "high",
                            "estimated_duration": "30 minutes",
                            "dependencies": []
                        },
                        {
                            "id": "fallback_task_2",
                            "title": "Implement solution",
                            "description": "Execute the planned approach",
                            "priority": "high",
                            "estimated_duration": "60 minutes",
                            "dependencies": ["fallback_task_1"]
                        }
                    ]
                }
                console.print("🔄 Using fallback task structure", style="yellow")
            
            # Create execution plan if it doesn't exist
            if "execution_plan" not in state or state["execution_plan"] is None:
                execution_plan = ExecutionPlan(goal=goal)
            else:
                execution_plan = state["execution_plan"]
            
            # Add tasks to the plan
            self._add_tasks_to_plan(task_data["tasks"], execution_plan)
            
            return {
                "execution_plan": execution_plan,
                "current_depth": current_depth + 1,
                "planning_complete": current_depth >= max_depth
            }
            
        except Exception as e:
            console.print(f"[red]Error in decomposition: {str(e)}[/red]")
            # Create a simple fallback plan
            execution_plan = ExecutionPlan(goal=goal)
            fallback_task = Task(
                title="Complete goal manually",
                description=f"Manual completion of: {goal}",
                priority=TaskPriority.HIGH,
                estimated_duration=60
            )
            execution_plan.add_task(fallback_task)
            
            return {
                "execution_plan": execution_plan,
                "planning_complete": True
            }
    
    def _add_tasks_to_plan(self, tasks_data: List[Dict], plan: ExecutionPlan, parent_id: Optional[str] = None):
        """Add tasks from JSON data to the execution plan."""
        for task_data in tasks_data:
            # Create task
            task = Task(
                title=task_data["title"],
                description=task_data["description"],
                priority=TaskPriority(task_data.get("priority", "medium")),
                estimated_duration=task_data.get("estimated_duration"),
                dependencies=task_data.get("dependencies", [])
            )
            
            # Add to plan
            plan.add_task(task, parent_id)
            
            # Add subtasks recursively
            if "subtasks" in task_data and task_data["subtasks"]:
                self._add_tasks_to_plan(task_data["subtasks"], plan, task.id)
    
    def execute_task(self, state: PlanningState) -> Dict[str, Any]:
        """Execute the next ready task."""
        execution_plan = state["execution_plan"]
        
        # Get next ready task
        ready_tasks = execution_plan.get_ready_tasks()
        
        if not ready_tasks:
            return {"planning_complete": True}
        
        # Execute the highest priority task
        task = ready_tasks[0]
        task.status = TaskStatus.IN_PROGRESS
        
        execution_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a task executor. Your job is to complete the given task and provide a detailed result.

Guidelines:
1. Understand the task requirements thoroughly
2. Provide a comprehensive solution or result
3. Be specific and actionable in your output
4. If the task cannot be completed, explain why and suggest alternatives"""),
            ("human", """Task: {title}

Description: {description}

Dependencies completed: {dependencies}

Please execute this task and provide the result.""")
        ])
        
        try:
            # Get dependency results
            dep_results = []
            for dep_id in task.dependencies:
                if dep_id in execution_plan.task_registry:
                    dep_task = execution_plan.task_registry[dep_id]
                    if dep_task.result:
                        dep_results.append(f"{dep_task.title}: {dep_task.result}")
            
            response = self.model.invoke(
                execution_prompt.format_messages(
                    title=task.title,
                    description=task.description,
                    dependencies="; ".join(dep_results) if dep_results else "None"
                )
            )
            
            # Mark task as completed
            execution_plan.mark_task_completed(task.id, response.content)
            
            return {
                "execution_plan": execution_plan,
                "current_task_id": task.id
            }
            
        except Exception as e:
            # Mark task as failed
            execution_plan.mark_task_failed(task.id, str(e))
            
            return {
                "execution_plan": execution_plan,
                "current_task_id": task.id
            }
    
    def evaluate_progress(self, state: PlanningState) -> Dict[str, Any]:
        """Evaluate the current progress and determine next steps."""
        execution_plan = state["execution_plan"]
        progress = execution_plan.get_progress_summary()
        
        return {
            "execution_plan": execution_plan,
            "planning_complete": progress["is_complete"]
        }
    
    def should_continue_execution(self, state: PlanningState) -> str:
        """Determine if execution should continue."""
        execution_plan = state["execution_plan"]
        
        if execution_plan.is_complete():
            return "complete"
        else:
            return "continue"
    
    def should_continue_planning(self, state: PlanningState) -> str:
        """Determine if planning should continue."""
        execution_plan = state["execution_plan"]
        planning_complete = state["planning_complete"]
        
        if execution_plan.is_complete():
            return "complete"
        elif planning_complete:
            return "execute_next"
        else:
            # Check if we need to replan (e.g., too many failures)
            progress = execution_plan.get_progress_summary()
            if progress["failed_tasks"] > progress["total_tasks"] * 0.3:  # 30% failure rate
                return "replan"
            else:
                return "execute_next"
    
    def create_plan(self, request: PlanningRequest) -> ExecutionPlan:
        """Create a hierarchical plan for the given goal."""
        try:
            initial_state = {
                "goal": request.goal,
                "context": request.context,
                "max_depth": request.max_depth,
                "max_tasks_per_level": request.max_tasks_per_level,
                "current_depth": 0,
                "execution_plan": None,
                "current_task_id": None,
                "planning_complete": False
            }
            
            # Run only the planning phase
            result = self.decompose_goal(initial_state)
            return result["execution_plan"]
            
        except Exception as e:
            console.print(f"[red]Error creating plan: {str(e)}[/red]")
            # Return empty plan
            return ExecutionPlan(goal=request.goal)
    
    def execute_plan(self, execution_plan: ExecutionPlan) -> ExecutionPlan:
        """Execute a hierarchical plan."""
        try:
            initial_state = {
                "goal": execution_plan.goal,
                "context": "",
                "max_depth": 3,
                "max_tasks_per_level": 5,
                "current_depth": 0,
                "execution_plan": execution_plan,
                "current_task_id": None,
                "planning_complete": True
            }
            
            # Run the execution graph
            final_state = self.graph.invoke(initial_state)
            return final_state["execution_plan"]
            
        except Exception as e:
            console.print(f"[red]Error executing plan: {str(e)}[/red]")
            return execution_plan


def main():
    """Demo the hierarchical planning agent."""
    console.print(Panel.fit("🏗️ Hierarchical Planning Pattern Demo", style="bold blue"))

    # Validate environment
    if not validate_environment():
        console.print("❌ Environment validation failed", style="bold red")
        return

    # Initialize planner
    console.print("Initializing hierarchical planner...", style="yellow")
    planner = HierarchicalPlanner()
    console.print("✅ Planner initialized successfully!", style="green")

    # Demo goals that benefit from hierarchical planning
    demo_goals = [
        {
            "goal": "Plan and organize a team retreat for 20 people",
            "context": "Budget: $5000, Duration: 2 days, Mix of work sessions and team building"
        },
        {
            "goal": "Launch a new mobile app",
            "context": "iOS and Android, Social media features, Target audience: young adults"
        },
        {
            "goal": "Write and publish a technical blog post about AI",
            "context": "Target audience: software developers, 2000-3000 words, include code examples"
        }
    ]

    console.print("\n🚀 Running hierarchical planning demos...", style="bold cyan")

    for i, demo in enumerate(demo_goals, 1):
        console.print(f"\n[bold]🎯 Demo {i}:[/bold] {demo['goal']}")
        console.print(f"[dim]Context: {demo['context']}[/dim]")
        console.print("=" * 80, style="dim")

        # Create planning request
        request = PlanningRequest(
            goal=demo["goal"],
            context=demo["context"],
            max_depth=3,
            max_tasks_per_level=4
        )

        # Create plan
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Creating hierarchical plan...", total=None)
            plan = planner.create_plan(request)

        # Display plan
        console.print("\n[bold green]📋 Generated Plan:[/bold green]")
        console.print(Panel(plan.get_task_hierarchy_display(), border_style="green"))

        # Show progress summary
        progress_summary = plan.get_progress_summary()
        console.print(f"\n[bold blue]📊 Plan Summary:[/bold blue]")
        console.print(f"  Total tasks: {progress_summary['total_tasks']}")
        console.print(f"  Estimated completion: {sum(t.estimated_duration or 0 for t in plan.task_registry.values() if not t.has_subtasks())} minutes")

        # Ask if user wants to execute the plan
        console.print(f"\n[yellow]Would you like to execute this plan? (y/n):[/yellow]", end=" ")
        try:
            choice = input().lower().strip()
            if choice == 'y':
                console.print("\n[bold yellow]🔄 Executing plan...[/bold yellow]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Executing tasks...", total=None)
                    executed_plan = planner.execute_plan(plan)

                # Display results
                console.print("\n[bold green]✅ Execution Complete![/bold green]")
                console.print(Panel(executed_plan.get_task_hierarchy_display(), border_style="green"))

                final_progress = executed_plan.get_progress_summary()
                console.print(f"\n[bold blue]📊 Final Results:[/bold blue]")
                console.print(f"  Completed: {final_progress['completed_tasks']}/{final_progress['total_tasks']}")
                console.print(f"  Success rate: {final_progress['progress_percentage']:.1f}%")
                console.print(f"  Failed tasks: {final_progress['failed_tasks']}")
            else:
                console.print("[dim]Skipping execution...[/dim]")
        except KeyboardInterrupt:
            console.print("\n[dim]Skipping execution...[/dim]")

    # Interactive mode
    console.print("\n🎯 Interactive Mode (type 'quit' to exit)", style="bold magenta")
    console.print("-" * 80, style="dim")

    while True:
        try:
            goal = console.input("\n[bold cyan]Enter your goal:[/bold cyan] ")
            if goal.lower() in ['quit', 'exit']:
                break

            context = console.input("[bold cyan]Enter context (optional):[/bold cyan] ")

            # Create and execute plan
            request = PlanningRequest(
                goal=goal,
                context=context if context else None,
                max_depth=3,
                max_tasks_per_level=4
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Planning...", total=None)
                plan = planner.create_plan(request)

            console.print("\n[bold green]📋 Your Plan:[/bold green]")
            console.print(Panel(plan.get_task_hierarchy_display(), border_style="green"))

            # Ask about execution
            execute = console.input("\n[yellow]Execute this plan? (y/n):[/yellow] ")
            if execute.lower() == 'y':
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Executing...", total=None)
                    executed_plan = planner.execute_plan(plan)

                console.print("\n[bold green]✅ Execution Results:[/bold green]")
                console.print(Panel(executed_plan.get_task_hierarchy_display(), border_style="green"))

        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
