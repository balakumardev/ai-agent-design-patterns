"""Multi-Agent Coordination Pattern Implementation using LangGraph."""

import os
import sys
import json
import asyncio
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
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime

# Add parent directory to path for shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utils import create_llm, validate_environment

from agent_models import (
    Agent, AgentRole, AgentCapability, AgentStatus, Message, MessageType,
    CoordinationTask, CoordinationPlan, CoordinationRequest
)

load_dotenv()

# Initialize Rich console for better output
console = Console()


class CoordinationState(TypedDict):
    """State of the multi-agent coordination graph."""
    goal: str
    context: Optional[str]
    coordination_plan: CoordinationPlan
    current_task_id: Optional[str]
    coordination_complete: bool
    iteration: int
    max_iterations: int


class MultiAgentCoordinator:
    """A LangGraph-based system for coordinating multiple AI agents."""
    
    def __init__(self, model_name: str = None):
        """Initialize the multi-agent coordinator."""
        self.model = create_llm(model_name=model_name, temperature=0.3)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for multi-agent coordination."""
        workflow = StateGraph(CoordinationState)
        
        # Add nodes
        workflow.add_node("setup_agents", self.setup_agents)
        workflow.add_node("plan_coordination", self.plan_coordination)
        workflow.add_node("execute_coordination", self.execute_coordination)
        workflow.add_node("monitor_progress", self.monitor_progress)
        
        # Set entry point
        workflow.set_entry_point("setup_agents")
        
        # Add edges
        workflow.add_edge("setup_agents", "plan_coordination")
        workflow.add_edge("plan_coordination", "execute_coordination")
        workflow.add_conditional_edges(
            "execute_coordination",
            self.should_continue_coordination,
            {
                "continue": "monitor_progress",
                "complete": END,
            },
        )
        workflow.add_conditional_edges(
            "monitor_progress",
            self.should_continue_monitoring,
            {
                "execute_next": "execute_coordination",
                "replan": "plan_coordination",
                "complete": END,
            },
        )
        
        return workflow.compile()
    
    def setup_agents(self, state: CoordinationState) -> Dict[str, Any]:
        """Set up the agents needed for the coordination task."""
        goal = state["goal"]
        context = state.get("context", "")
        
        setup_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in multi-agent system design. Your job is to determine what types of agents are needed for a given goal and set up their capabilities.

Guidelines:
1. Identify the key roles needed (researcher, analyst, writer, reviewer, coordinator, etc.)
2. Define specific capabilities for each agent
3. Consider the workflow and how agents will collaborate
4. Limit to 3-5 agents for manageable coordination

Return your response as a JSON object with this structure:
{{
    "agents": [
        {{
            "name": "Agent name",
            "role": "researcher|analyst|writer|reviewer|coordinator|executor|specialist",
            "capabilities": [
                {{
                    "name": "capability_name",
                    "description": "What this capability does",
                    "input_types": ["text", "data", "query"],
                    "output_types": ["analysis", "report", "summary"],
                    "estimated_duration": 300
                }}
            ]
        }}
    ]
}}"""),
            ("human", """Goal: {goal}
            
Context: {context}

Please design a multi-agent system to accomplish this goal.""")
        ])
        
        try:
            response = self.model.invoke(
                setup_prompt.format_messages(goal=goal, context=context)
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
                agent_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                console.print(f"❌ JSON parsing failed in agent setup: {e}", style="red")
                console.print(f"Raw response: {response_text[:200]}...", style="dim")

                # Fallback: create a simple agent structure
                agent_data = {
                    "agents": [
                        {
                            "id": "coordinator",
                            "role": "Coordinator",
                            "capabilities": ["task_coordination"],
                            "description": "Coordinates tasks and manages workflow"
                        }
                    ]
                }
                console.print("🔄 Using fallback agent structure", style="yellow")
            
            # Create coordination plan
            plan = CoordinationPlan(goal=goal)
            
            # Create agents
            for agent_info in agent_data["agents"]:
                agent = Agent(
                    name=agent_info["name"],
                    role=AgentRole(agent_info["role"]),
                    capabilities=[
                        AgentCapability(
                            name=cap["name"],
                            description=cap["description"],
                            input_types=cap["input_types"],
                            output_types=cap["output_types"],
                            estimated_duration=cap.get("estimated_duration", 300)
                        )
                        for cap in agent_info["capabilities"]
                    ]
                )
                plan.add_agent(agent)
            
            return {
                "coordination_plan": plan,
                "iteration": 1
            }
            
        except Exception as e:
            console.print(f"[red]Error in agent setup: {str(e)}[/red]")
            # Create a simple fallback plan
            plan = CoordinationPlan(goal=goal)
            
            # Add a basic coordinator agent
            coordinator = Agent(
                name="Coordinator",
                role=AgentRole.COORDINATOR,
                capabilities=[
                    AgentCapability(
                        name="task_coordination",
                        description="Coordinate tasks between agents",
                        input_types=["goal"],
                        output_types=["plan"],
                        estimated_duration=300
                    )
                ]
            )
            plan.add_agent(coordinator)
            
            return {
                "coordination_plan": plan,
                "iteration": 1
            }
    
    def plan_coordination(self, state: CoordinationState) -> Dict[str, Any]:
        """Plan the coordination strategy and create tasks."""
        goal = state["goal"]
        plan = state["coordination_plan"]
        
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a coordination planner. Your job is to create a sequence of tasks that the available agents can execute to achieve the goal.

Guidelines:
1. Break down the goal into specific tasks
2. Assign tasks to appropriate agents based on their capabilities
3. Consider dependencies between tasks
4. Create a logical workflow

Available agents and their capabilities:
{agent_info}

Return your response as a JSON object with this structure:
{{
    "tasks": [
        {{
            "title": "Task title",
            "description": "Detailed task description",
            "required_capabilities": ["capability1", "capability2"],
            "assigned_agents": ["agent_id1"],
            "dependencies": ["task_id1", "task_id2"]
        }}
    ]
}}"""),
            ("human", """Goal: {goal}

Please create a coordination plan with specific tasks for the available agents.""")
        ])
        
        try:
            # Prepare agent information
            agent_info = []
            for agent in plan.agents.values():
                caps = [f"{cap.name}: {cap.description}" for cap in agent.capabilities]
                agent_info.append(f"{agent.name} ({agent.role.value}): {', '.join(caps)}")
            
            response = self.model.invoke(
                planning_prompt.format_messages(
                    goal=goal,
                    agent_info="\n".join(agent_info)
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
                console.print(f"❌ JSON parsing failed in coordination planning: {e}", style="red")
                console.print(f"Raw response: {response_text[:200]}...", style="dim")

                # Fallback: create a simple task structure
                task_data = {
                    "tasks": [
                        {
                            "title": "Complete goal",
                            "description": f"Work towards achieving: {goal}",
                            "required_capabilities": ["task_coordination"],
                            "assigned_agents": ["coordinator"],
                            "dependencies": []
                        }
                    ]
                }
                console.print("🔄 Using fallback task structure", style="yellow")
            
            # Create tasks
            for task_info in task_data["tasks"]:
                task = CoordinationTask(
                    title=task_info["title"],
                    description=task_info["description"],
                    required_capabilities=task_info["required_capabilities"]
                )
                
                # Assign agents based on capabilities
                for agent in plan.agents.values():
                    if any(agent.can_handle_task(cap) for cap in task.required_capabilities):
                        task.assign_agent(agent.id)
                        break
                
                plan.add_task(task)
            
            return {"coordination_plan": plan}
            
        except Exception as e:
            console.print(f"[red]Error in coordination planning: {str(e)}[/red]")
            # Create a simple fallback task
            fallback_task = CoordinationTask(
                title="Complete goal",
                description=f"Work together to complete: {goal}",
                required_capabilities=["task_coordination"]
            )
            
            # Assign to first available agent
            if plan.agents:
                first_agent = next(iter(plan.agents.values()))
                fallback_task.assign_agent(first_agent.id)
            
            plan.add_task(fallback_task)
            
            return {"coordination_plan": plan}
    
    def execute_coordination(self, state: CoordinationState) -> Dict[str, Any]:
        """Execute the next task in the coordination plan."""
        plan = state["coordination_plan"]
        
        # Find next task to execute
        next_task = None
        for task in plan.tasks:
            if task.status == "pending" and task.assigned_agents:
                next_task = task
                break
        
        if not next_task:
            return {"coordination_complete": True}
        
        # Execute the task
        next_task.status = "in_progress"
        assigned_agent_id = next_task.assigned_agents[0]
        assigned_agent = plan.agents[assigned_agent_id]
        assigned_agent.status = AgentStatus.BUSY
        assigned_agent.current_task = next_task.id
        
        execution_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are {agent_name}, a {agent_role} agent. Your job is to execute the assigned task using your capabilities.

Your capabilities:
{capabilities}

Guidelines:
1. Focus on your specific role and expertise
2. Provide detailed, actionable results
3. Consider how your work fits into the larger goal
4. Communicate clearly with other agents if needed"""),
            ("human", """Task: {task_title}

Description: {task_description}

Goal context: {goal}

Please execute this task and provide the result.""")
        ])
        
        try:
            # Prepare capability information
            caps = [f"- {cap.name}: {cap.description}" for cap in assigned_agent.capabilities]
            
            response = self.model.invoke(
                execution_prompt.format_messages(
                    agent_name=assigned_agent.name,
                    agent_role=assigned_agent.role.value,
                    capabilities="\n".join(caps),
                    task_title=next_task.title,
                    task_description=next_task.description,
                    goal=plan.goal
                )
            )
            
            # Mark task as completed
            next_task.status = "completed"
            next_task.result = response.content
            next_task.completed_at = datetime.now()
            plan.completed_tasks.add(next_task.id)
            
            # Update agent status
            assigned_agent.status = AgentStatus.IDLE
            assigned_agent.current_task = None
            
            # Create a message about task completion
            completion_message = Message(
                sender_id=assigned_agent.id,
                recipient_id="ALL",
                message_type=MessageType.STATUS_UPDATE,
                content=f"Completed task: {next_task.title}. Result: {response.content[:200]}..."
            )
            plan.send_message(completion_message)
            
            return {
                "coordination_plan": plan,
                "current_task_id": next_task.id
            }
            
        except Exception as e:
            # Mark task as failed
            next_task.status = "failed"
            next_task.result = f"FAILED: {str(e)}"
            plan.failed_tasks.add(next_task.id)
            
            # Update agent status
            assigned_agent.status = AgentStatus.ERROR
            assigned_agent.current_task = None
            
            return {
                "coordination_plan": plan,
                "current_task_id": next_task.id
            }
    
    def monitor_progress(self, state: CoordinationState) -> Dict[str, Any]:
        """Monitor the progress of coordination and determine next steps."""
        plan = state["coordination_plan"]
        iteration = state["iteration"]
        
        progress = plan.get_progress_summary()
        
        return {
            "coordination_plan": plan,
            "coordination_complete": progress["is_complete"],
            "iteration": iteration + 1
        }
    
    def should_continue_coordination(self, state: CoordinationState) -> str:
        """Determine if coordination should continue."""
        plan = state["coordination_plan"]
        progress = plan.get_progress_summary()
        
        if progress["is_complete"]:
            return "complete"
        else:
            return "continue"
    
    def should_continue_monitoring(self, state: CoordinationState) -> str:
        """Determine if monitoring should continue."""
        plan = state["coordination_plan"]
        iteration = state["iteration"]
        max_iterations = state.get("max_iterations", 10)
        
        progress = plan.get_progress_summary()
        
        if progress["is_complete"]:
            return "complete"
        elif iteration >= max_iterations:
            return "complete"
        else:
            return "execute_next"

    def coordinate(self, request: CoordinationRequest) -> CoordinationPlan:
        """Coordinate multiple agents to achieve a goal."""
        try:
            initial_state = {
                "goal": request.goal,
                "context": request.context,
                "coordination_plan": None,
                "current_task_id": None,
                "coordination_complete": False,
                "iteration": 0,
                "max_iterations": 10
            }

            # Run the coordination graph
            final_state = self.graph.invoke(initial_state)
            return final_state["coordination_plan"]

        except Exception as e:
            console.print(f"[red]Error in coordination: {str(e)}[/red]")
            # Return empty plan
            return CoordinationPlan(goal=request.goal)


def main():
    """Demo the multi-agent coordination system."""
    console.print(Panel.fit("🤝 Multi-Agent Coordination Pattern Demo", style="bold blue"))

    # Validate environment
    if not validate_environment():
        console.print("❌ Environment validation failed", style="bold red")
        return

    # Initialize coordinator
    console.print("Initializing multi-agent coordinator...", style="yellow")
    coordinator = MultiAgentCoordinator()
    console.print("✅ Coordinator initialized successfully!", style="green")

    # Demo goals that benefit from multi-agent coordination
    demo_goals = [
        {
            "goal": "Create a comprehensive market research report for a new AI product",
            "context": "Target market: enterprise software, competitors: 5 major players, timeline: 2 weeks",
            "required_roles": ["researcher", "analyst", "writer", "reviewer"]
        },
        {
            "goal": "Develop a content marketing strategy for a tech startup",
            "context": "B2B SaaS company, target audience: CTOs and engineering managers, budget: $50k",
            "required_roles": ["researcher", "strategist", "writer", "coordinator"]
        },
        {
            "goal": "Plan and execute a software architecture review",
            "context": "Microservices architecture, 15 services, performance issues, security concerns",
            "required_roles": ["architect", "security_specialist", "performance_analyst", "reviewer"]
        }
    ]

    console.print("\n🚀 Running multi-agent coordination demos...", style="bold cyan")

    for i, demo in enumerate(demo_goals, 1):
        console.print(f"\n[bold]🎯 Demo {i}:[/bold] {demo['goal']}")
        console.print(f"[dim]Context: {demo['context']}[/dim]")
        console.print(f"[dim]Required roles: {', '.join(demo['required_roles'])}[/dim]")
        console.print("=" * 80, style="dim")

        # Create coordination request
        request = CoordinationRequest(
            goal=demo["goal"],
            context=demo["context"],
            required_roles=demo["required_roles"],
            max_agents=4
        )

        # Execute coordination
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Coordinating agents...", total=None)
            plan = coordinator.coordinate(request)

        # Display results
        console.print("\n[bold green]🤖 Agent Team:[/bold green]")
        agent_table = Table(show_header=True, header_style="bold magenta")
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Role", style="green")
        agent_table.add_column("Capabilities", style="yellow")
        agent_table.add_column("Status", style="blue")

        for agent in plan.agents.values():
            caps = ", ".join([cap.name for cap in agent.capabilities[:2]])
            if len(agent.capabilities) > 2:
                caps += f" (+{len(agent.capabilities)-2} more)"

            status_emoji = {
                "idle": "💤",
                "busy": "⚡",
                "waiting": "⏳",
                "error": "❌",
                "offline": "📴"
            }

            agent_table.add_row(
                agent.name,
                agent.role.value.title(),
                caps,
                f"{status_emoji.get(agent.status.value, '❓')} {agent.status.value.title()}"
            )

        console.print(agent_table)

        # Display tasks and results
        console.print(f"\n[bold green]📋 Coordination Tasks:[/bold green]")
        task_table = Table(show_header=True, header_style="bold magenta")
        task_table.add_column("Task", style="cyan")
        task_table.add_column("Assigned Agent", style="green")
        task_table.add_column("Status", style="yellow")
        task_table.add_column("Result Preview", style="blue")

        for task in plan.tasks:
            assigned_agent_name = "Unassigned"
            if task.assigned_agents:
                agent_id = task.assigned_agents[0]
                if agent_id in plan.agents:
                    assigned_agent_name = plan.agents[agent_id].name

            status_emoji = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌"
            }

            result_preview = ""
            if task.result:
                result_preview = task.result[:50] + "..." if len(task.result) > 50 else task.result

            task_table.add_row(
                task.title,
                assigned_agent_name,
                f"{status_emoji.get(task.status, '❓')} {task.status.title()}",
                result_preview
            )

        console.print(task_table)

        # Show coordination summary
        progress_summary = plan.get_progress_summary()
        agent_status_summary = plan.get_agent_status_summary()

        console.print(f"\n[bold blue]📊 Coordination Summary:[/bold blue]")
        console.print(f"  Tasks completed: {progress_summary['completed_tasks']}/{progress_summary['total_tasks']}")
        console.print(f"  Success rate: {progress_summary['progress_percentage']:.1f}%")
        console.print(f"  Agent statuses: {dict(agent_status_summary)}")
        console.print(f"  Messages exchanged: {len(plan.message_queue)}")

        # Show recent messages
        if plan.message_queue:
            console.print(f"\n[bold yellow]💬 Recent Messages:[/bold yellow]")
            for msg in plan.message_queue[-3:]:  # Show last 3 messages
                sender_name = plan.agents.get(msg.sender_id, type('obj', (object,), {'name': msg.sender_id})).name
                console.print(f"  📨 {sender_name}: {msg.content[:80]}...")

    # Interactive mode
    console.print("\n🎯 Interactive Mode (type 'quit' to exit)", style="bold magenta")
    console.print("-" * 80, style="dim")

    while True:
        try:
            goal = console.input("\n[bold cyan]Enter coordination goal:[/bold cyan] ")
            if goal.lower() in ['quit', 'exit']:
                break

            context = console.input("[bold cyan]Enter context (optional):[/bold cyan] ")

            # Create and execute coordination
            request = CoordinationRequest(
                goal=goal,
                context=context if context else None,
                max_agents=4
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Coordinating...", total=None)
                plan = coordinator.coordinate(request)

            console.print("\n[bold green]🤖 Your Agent Team:[/bold green]")
            for agent in plan.agents.values():
                console.print(f"  • {agent.name} ({agent.role.value}) - {len(agent.capabilities)} capabilities")

            console.print(f"\n[bold green]📋 Coordination Results:[/bold green]")
            for task in plan.tasks:
                status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌"}
                console.print(f"  {status_emoji.get(task.status, '❓')} {task.title}")
                if task.result:
                    console.print(f"    Result: {task.result[:100]}...")

        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
