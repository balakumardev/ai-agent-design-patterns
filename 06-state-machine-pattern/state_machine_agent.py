"""State Machine Pattern Implementation for Predictable Agent Behavior."""

import os
import sys
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Add parent directory to path for shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utils import create_llm, validate_environment
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

from state_models import (
    State, StateContext, StateMachineDefinition, Transition, TransitionCondition,
    StateType, InitialState, FinalState, ErrorState, ProcessingState, 
    DecisionState, WaitingState
)

load_dotenv()

# Initialize Rich console for better output
console = Console()


@dataclass
class ExecutionResult:
    """Result of state machine execution."""
    final_state: str
    context: StateContext
    execution_path: List[str]
    success: bool
    error_message: str = ""


class StateMachineAgent:
    """An agent that follows a predictable state machine pattern."""
    
    def __init__(self, model_name: str = None):
        """Initialize the state machine agent."""
        self.model = create_llm(model_name=model_name, temperature=0.1)
        self.state_machines: Dict[str, StateMachineDefinition] = {}
        self.current_execution: Optional[ExecutionResult] = None
    
    def register_state_machine(self, definition: StateMachineDefinition):
        """Register a state machine definition."""
        # Validate the state machine
        errors = definition.validate()
        if errors:
            raise ValueError(f"Invalid state machine: {errors}")
        
        self.state_machines[definition.name] = definition
        console.print(f"✅ Registered state machine: {definition.name}", style="green")
    
    def execute_state_machine(
        self, 
        machine_name: str, 
        initial_context: Optional[StateContext] = None,
        trace_execution: bool = True
    ) -> ExecutionResult:
        """Execute a state machine with the given context."""
        if machine_name not in self.state_machines:
            raise ValueError(f"State machine '{machine_name}' not found")
        
        definition = self.state_machines[machine_name]
        context = initial_context or StateContext()
        execution_path = []
        current_state_name = definition.initial_state
        
        if trace_execution:
            console.print(f"🚀 Executing state machine: {machine_name}", style="bold cyan")
        
        try:
            while True:
                # Check for timeout or max iterations
                if context.is_timeout():
                    context.error_message = "Execution timeout"
                    current_state_name = self._find_error_state(definition)
                    break
                
                if context.is_max_iterations_reached():
                    context.error_message = "Maximum iterations reached"
                    current_state_name = self._find_error_state(definition)
                    break
                
                # Get current state
                if current_state_name not in definition.states:
                    context.error_message = f"State '{current_state_name}' not found"
                    current_state_name = self._find_error_state(definition)
                    break
                
                current_state = definition.states[current_state_name]
                execution_path.append(current_state_name)
                
                if trace_execution:
                    console.print(f"📍 Current state: {current_state_name}", style="yellow")
                
                # Execute state entry actions
                current_state.on_entry(context)
                
                # Execute state logic
                context = current_state.execute(context)
                context.increment_iteration()
                
                # Execute state exit actions
                current_state.on_exit(context)
                
                # Check if we're in a final state
                if definition.is_final_state(current_state_name) or definition.is_error_state(current_state_name):
                    break
                
                # Determine next state
                next_state = definition.get_next_state(current_state_name, context)
                
                if next_state is None:
                    context.error_message = f"No valid transition from state '{current_state_name}'"
                    current_state_name = self._find_error_state(definition)
                    break
                
                current_state_name = next_state
            
            # Create execution result
            success = definition.is_final_state(current_state_name) and not context.error_message
            
            result = ExecutionResult(
                final_state=current_state_name,
                context=context,
                execution_path=execution_path,
                success=success,
                error_message=context.error_message
            )
            
            self.current_execution = result
            
            if trace_execution:
                self._display_execution_result(result, definition)
            
            return result
            
        except Exception as e:
            error_result = ExecutionResult(
                final_state="error",
                context=context,
                execution_path=execution_path,
                success=False,
                error_message=str(e)
            )
            
            if trace_execution:
                console.print(f"❌ Execution error: {str(e)}", style="bold red")
            
            return error_result
    
    def _find_error_state(self, definition: StateMachineDefinition) -> str:
        """Find an appropriate error state."""
        if definition.error_states:
            return next(iter(definition.error_states))
        else:
            # Create a temporary error state if none exists
            return "error"
    
    def _display_execution_result(self, result: ExecutionResult, definition: StateMachineDefinition):
        """Display the execution result in a formatted way."""
        if result.success:
            console.print("✅ State machine execution completed successfully", style="bold green")
        else:
            console.print(f"❌ State machine execution failed: {result.error_message}", style="bold red")
        
        # Display execution path
        console.print(f"\n📊 Execution Summary:", style="bold blue")
        
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Final State", result.final_state)
        summary_table.add_row("States Visited", str(len(result.execution_path)))
        summary_table.add_row("Iterations", str(result.context.iteration_count))
        summary_table.add_row("Execution Time", f"{result.context.get_elapsed_time():.2f}s")
        summary_table.add_row("Success", "✅ Yes" if result.success else "❌ No")
        
        console.print(summary_table)
        
        # Display execution path
        if result.execution_path:
            console.print(f"\n🛤️ Execution Path:", style="bold blue")
            path_str = " → ".join(result.execution_path)
            console.print(Panel(path_str, border_style="blue"))
        
        # Display context data if available
        if result.context.current_data:
            console.print(f"\n📋 Final Context Data:", style="bold blue")
            for key, value in result.context.current_data.items():
                console.print(f"  {key}: {value}")
    
    def create_llm_processing_state(
        self, 
        name: str, 
        system_prompt: str, 
        description: str = ""
    ) -> ProcessingState:
        """Create a processing state that uses the LLM."""
        
        def llm_processor(context: StateContext) -> StateContext:
            """Process using LLM."""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{user_input}")
            ])
            
            try:
                response = self.model.invoke(
                    prompt.format_messages(user_input=context.user_input)
                )
                
                context.current_data[f'{name}_response'] = response.content
                context.add_to_history(f"LLM processing in {name}: Generated response")
                
            except Exception as e:
                context.error_message = f"LLM processing error in {name}: {str(e)}"
            
            return context
        
        return ProcessingState(name, description, llm_processor)
    
    def create_llm_decision_state(
        self, 
        name: str, 
        system_prompt: str, 
        decision_options: List[str],
        description: str = ""
    ) -> DecisionState:
        """Create a decision state that uses the LLM."""
        
        def llm_decision(context: StateContext) -> str:
            """Make decision using LLM."""
            options_str = ", ".join(decision_options)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{system_prompt}\n\nYou must respond with exactly one of these options: {options_str}"),
                ("human", "{user_input}")
            ])
            
            try:
                response = self.model.invoke(
                    prompt.format_messages(user_input=context.user_input)
                )
                
                decision = response.content.strip().lower()
                
                # Validate decision
                if decision not in [opt.lower() for opt in decision_options]:
                    # Default to first option if invalid
                    decision = decision_options[0].lower()
                
                return decision
                
            except Exception as e:
                context.error_message = f"LLM decision error in {name}: {str(e)}"
                return decision_options[0].lower()  # Default option
        
        return DecisionState(name, description, llm_decision)
    
    def visualize_state_machine(self, machine_name: str, save_path: Optional[str] = None):
        """Visualize a state machine using Graphviz."""
        if machine_name not in self.state_machines:
            console.print(f"State machine '{machine_name}' not found", style="red")
            return
        
        definition = self.state_machines[machine_name]
        
        try:
            import graphviz
            
            dot_source = definition.to_graphviz()
            
            if save_path:
                # Save to file
                graph = graphviz.Source(dot_source)
                graph.render(save_path, format='png', cleanup=True)
                console.print(f"✅ State machine diagram saved to {save_path}.png", style="green")
            else:
                # Display in console (text representation)
                console.print(f"\n🔄 State Machine: {machine_name}", style="bold blue")
                console.print("States and Transitions:", style="cyan")
                
                # Create a tree view
                tree = Tree(f"[bold]{machine_name}[/bold]")
                
                states_node = tree.add("[cyan]States[/cyan]")
                for state_name, state in definition.states.items():
                    state_style = "green" if state.state_type.value == "initial" else \
                                 "blue" if state.state_type.value == "final" else \
                                 "red" if state.state_type.value == "error" else "white"
                    states_node.add(f"[{state_style}]{state_name}[/{state_style}] ({state.state_type.value})")
                
                transitions_node = tree.add("[yellow]Transitions[/yellow]")
                for transition in definition.transitions:
                    transitions_node.add(f"{transition.from_state} → {transition.to_state} ({transition.condition.value})")
                
                console.print(tree)
                
        except ImportError:
            console.print("Graphviz not available. Install with: pip install graphviz", style="yellow")
            
            # Fallback to text representation
            console.print(f"\n🔄 State Machine: {machine_name}", style="bold blue")
            console.print(definition.to_graphviz())
    
    def get_state_machine_info(self, machine_name: str) -> Dict[str, Any]:
        """Get information about a state machine."""
        if machine_name not in self.state_machines:
            return {}
        
        definition = self.state_machines[machine_name]
        
        return {
            "name": definition.name,
            "description": definition.description,
            "states": {name: {
                "type": state.state_type.value,
                "description": state.description
            } for name, state in definition.states.items()},
            "transitions": [{
                "from": t.from_state,
                "to": t.to_state,
                "condition": t.condition.value,
                "description": t.description
            } for t in definition.transitions],
            "initial_state": definition.initial_state,
            "final_states": list(definition.final_states),
            "error_states": list(definition.error_states)
        }
    
    def list_state_machines(self):
        """List all registered state machines."""
        if not self.state_machines:
            console.print("No state machines registered", style="yellow")
            return
        
        console.print("📋 Registered State Machines:", style="bold blue")
        
        machines_table = Table(show_header=True, header_style="bold magenta")
        machines_table.add_column("Name", style="cyan")
        machines_table.add_column("Description", style="green")
        machines_table.add_column("States", style="yellow")
        machines_table.add_column("Transitions", style="blue")
        
        for name, definition in self.state_machines.items():
            machines_table.add_row(
                name,
                definition.description,
                str(len(definition.states)),
                str(len(definition.transitions))
            )
        
        console.print(machines_table)


def create_customer_support_state_machine(agent: StateMachineAgent) -> StateMachineDefinition:
    """Create a customer support state machine example."""

    # Create states
    initial = InitialState("start", "Customer support session started")

    # Greeting state
    greeting = agent.create_llm_processing_state(
        "greeting",
        "You are a friendly customer support agent. Greet the customer and ask how you can help them today. Be warm and professional.",
        "Greet the customer and gather initial information"
    )

    # Intent classification state
    classify_intent = agent.create_llm_decision_state(
        "classify_intent",
        "Analyze the customer's message and classify their intent. Consider the context and determine what type of help they need.",
        ["technical_issue", "billing_question", "general_inquiry", "complaint"],
        "Classify customer intent"
    )

    # Technical support state
    technical_support = agent.create_llm_processing_state(
        "technical_support",
        "You are a technical support specialist. Help the customer resolve their technical issue. Ask clarifying questions if needed and provide step-by-step solutions.",
        "Handle technical issues"
    )

    # Billing support state
    billing_support = agent.create_llm_processing_state(
        "billing_support",
        "You are a billing specialist. Help the customer with their billing questions. Be clear about charges, refunds, and payment processes.",
        "Handle billing questions"
    )

    # General inquiry state
    general_inquiry = agent.create_llm_processing_state(
        "general_inquiry",
        "You are a helpful customer service representative. Answer the customer's general questions about products, services, or policies.",
        "Handle general inquiries"
    )

    # Complaint handling state
    complaint_handling = agent.create_llm_processing_state(
        "complaint_handling",
        "You are a customer service manager. Handle the customer's complaint with empathy and professionalism. Acknowledge their concerns and offer solutions.",
        "Handle customer complaints"
    )

    # Satisfaction check state
    satisfaction_check = agent.create_llm_decision_state(
        "satisfaction_check",
        "Ask the customer if their issue has been resolved to their satisfaction. Based on their response, determine if they are satisfied or need further assistance.",
        ["satisfied", "needs_more_help"],
        "Check customer satisfaction"
    )

    # Escalation state
    escalation = agent.create_llm_processing_state(
        "escalation",
        "The customer needs additional help. Escalate to a supervisor or specialist. Explain the escalation process and ensure smooth handoff.",
        "Escalate to higher level support"
    )

    # Closing state
    closing = agent.create_llm_processing_state(
        "closing",
        "Thank the customer for contacting support. Provide any final information and close the session professionally.",
        "Close the support session"
    )

    final = FinalState("end", "Customer support session completed")
    error = ErrorState("error", "Error in customer support process")

    # Create state machine definition
    definition = StateMachineDefinition(
        name="customer_support",
        description="Customer support workflow with intent classification and specialized handling",
        initial_state="start"
    )

    # Add states
    for state in [initial, greeting, classify_intent, technical_support, billing_support,
                  general_inquiry, complaint_handling, satisfaction_check, escalation, closing, final, error]:
        definition.add_state(state)

    # Add transitions
    definition.add_transition(Transition("start", "greeting", TransitionCondition.ALWAYS))
    definition.add_transition(Transition("greeting", "classify_intent", TransitionCondition.ON_SUCCESS))

    # Intent-based routing
    definition.add_transition(Transition("classify_intent", "technical_support", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('classify_intent_response', '').lower().find('technical') != -1))
    definition.add_transition(Transition("classify_intent", "billing_support", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('classify_intent_response', '').lower().find('billing') != -1))
    definition.add_transition(Transition("classify_intent", "general_inquiry", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('classify_intent_response', '').lower().find('general') != -1))
    definition.add_transition(Transition("classify_intent", "complaint_handling", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('classify_intent_response', '').lower().find('complaint') != -1))

    # From support states to satisfaction check
    for support_state in ["technical_support", "billing_support", "general_inquiry", "complaint_handling"]:
        definition.add_transition(Transition(support_state, "satisfaction_check", TransitionCondition.ON_SUCCESS))

    # Satisfaction-based routing
    definition.add_transition(Transition("satisfaction_check", "closing", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('satisfaction_check_response', '').lower().find('satisfied') != -1))
    definition.add_transition(Transition("satisfaction_check", "escalation", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('satisfaction_check_response', '').lower().find('more_help') != -1))

    # From escalation back to satisfaction check
    definition.add_transition(Transition("escalation", "satisfaction_check", TransitionCondition.ON_SUCCESS))

    # Final transitions
    definition.add_transition(Transition("closing", "end", TransitionCondition.ON_SUCCESS))

    # Error transitions
    for state_name in definition.states.keys():
        if state_name not in ["error", "end"]:
            definition.add_transition(Transition(state_name, "error", TransitionCondition.ON_FAILURE))

    return definition


def create_task_planning_state_machine(agent: StateMachineAgent) -> StateMachineDefinition:
    """Create a task planning state machine example."""

    # Create states
    initial = InitialState("start", "Task planning session started")

    # Task analysis state
    analyze_task = agent.create_llm_processing_state(
        "analyze_task",
        "Analyze the given task. Break it down into key components, identify requirements, and assess complexity. Provide a structured analysis.",
        "Analyze and understand the task"
    )

    # Complexity assessment state
    assess_complexity = agent.create_llm_decision_state(
        "assess_complexity",
        "Based on the task analysis, assess the complexity level. Consider factors like time required, resources needed, and difficulty.",
        ["simple", "moderate", "complex"],
        "Assess task complexity"
    )

    # Simple task planning
    simple_planning = agent.create_llm_processing_state(
        "simple_planning",
        "Create a straightforward plan for this simple task. Provide clear, actionable steps that can be completed quickly.",
        "Plan simple task"
    )

    # Moderate task planning
    moderate_planning = agent.create_llm_processing_state(
        "moderate_planning",
        "Create a detailed plan for this moderate complexity task. Break it into phases, identify dependencies, and estimate timeframes.",
        "Plan moderate complexity task"
    )

    # Complex task planning
    complex_planning = agent.create_llm_processing_state(
        "complex_planning",
        "Create a comprehensive plan for this complex task. Include detailed phases, risk assessment, resource allocation, and contingency planning.",
        "Plan complex task"
    )

    # Review and validation
    review_plan = agent.create_llm_decision_state(
        "review_plan",
        "Review the created plan for completeness and feasibility. Check if any adjustments are needed.",
        ["approved", "needs_revision"],
        "Review and validate the plan"
    )

    # Plan revision
    revise_plan = agent.create_llm_processing_state(
        "revise_plan",
        "Revise the plan based on the review feedback. Address any gaps or issues identified during review.",
        "Revise the plan"
    )

    # Finalization
    finalize_plan = agent.create_llm_processing_state(
        "finalize_plan",
        "Finalize the task plan. Summarize key milestones, deliverables, and next steps. Ensure the plan is ready for execution.",
        "Finalize the task plan"
    )

    final = FinalState("end", "Task planning completed")
    error = ErrorState("error", "Error in task planning process")

    # Create state machine definition
    definition = StateMachineDefinition(
        name="task_planning",
        description="Intelligent task planning with complexity-based routing",
        initial_state="start"
    )

    # Add states
    for state in [initial, analyze_task, assess_complexity, simple_planning, moderate_planning,
                  complex_planning, review_plan, revise_plan, finalize_plan, final, error]:
        definition.add_state(state)

    # Add transitions
    definition.add_transition(Transition("start", "analyze_task", TransitionCondition.ALWAYS))
    definition.add_transition(Transition("analyze_task", "assess_complexity", TransitionCondition.ON_SUCCESS))

    # Complexity-based routing
    definition.add_transition(Transition("assess_complexity", "simple_planning", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('assess_complexity_response', '').lower().find('simple') != -1))
    definition.add_transition(Transition("assess_complexity", "moderate_planning", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('assess_complexity_response', '').lower().find('moderate') != -1))
    definition.add_transition(Transition("assess_complexity", "complex_planning", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('assess_complexity_response', '').lower().find('complex') != -1))

    # From planning states to review
    for planning_state in ["simple_planning", "moderate_planning", "complex_planning"]:
        definition.add_transition(Transition(planning_state, "review_plan", TransitionCondition.ON_SUCCESS))

    # Review-based routing
    definition.add_transition(Transition("review_plan", "finalize_plan", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('review_plan_response', '').lower().find('approved') != -1))
    definition.add_transition(Transition("review_plan", "revise_plan", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('review_plan_response', '').lower().find('revision') != -1))

    # From revision back to review
    definition.add_transition(Transition("revise_plan", "review_plan", TransitionCondition.ON_SUCCESS))

    # Final transition
    definition.add_transition(Transition("finalize_plan", "end", TransitionCondition.ON_SUCCESS))

    # Error transitions
    for state_name in definition.states.keys():
        if state_name not in ["error", "end"]:
            definition.add_transition(Transition(state_name, "error", TransitionCondition.ON_FAILURE))

    return definition


def main():
    """Demo the state machine agent."""
    console.print(Panel.fit("🔄 State Machine Pattern Demo", style="bold blue"))

    # Validate environment
    if not validate_environment():
        console.print("❌ Environment validation failed", style="bold red")
        return

    # Initialize agent
    console.print("Initializing state machine agent...", style="yellow")
    agent = StateMachineAgent()
    console.print("✅ Agent initialized successfully!", style="green")

    # Create and register state machines
    console.print("\n🏗️ Creating demo state machines...", style="cyan")

    # Customer support state machine
    customer_support_sm = create_customer_support_state_machine(agent)
    agent.register_state_machine(customer_support_sm)

    # Task planning state machine
    task_planning_sm = create_task_planning_state_machine(agent)
    agent.register_state_machine(task_planning_sm)

    # List registered state machines
    console.print("\n📋 Registered State Machines:", style="bold blue")
    agent.list_state_machines()

    # Demo scenarios
    demo_scenarios = [
        {
            "machine": "customer_support",
            "scenario": "Technical Issue",
            "input": "Hi, I'm having trouble logging into my account. The password reset isn't working.",
            "description": "Customer with a technical login issue"
        },
        {
            "machine": "customer_support",
            "scenario": "Billing Question",
            "input": "I was charged twice for my subscription this month. Can you help me understand why?",
            "description": "Customer with a billing inquiry"
        },
        {
            "machine": "task_planning",
            "scenario": "Simple Task",
            "input": "I need to write a thank you email to my team for completing a project.",
            "description": "Simple task planning scenario"
        },
        {
            "machine": "task_planning",
            "scenario": "Complex Task",
            "input": "I need to plan and execute a complete website redesign for our company, including user research, design, development, and launch.",
            "description": "Complex task planning scenario"
        }
    ]

    console.print("\n🚀 Running state machine demos...", style="bold cyan")

    for i, scenario in enumerate(demo_scenarios, 1):
        console.print(f"\n[bold]Demo {i}: {scenario['scenario']}[/bold]")
        console.print(f"[dim]Machine: {scenario['machine']}[/dim]")
        console.print(f"[dim]Description: {scenario['description']}[/dim]")
        console.print(f"[cyan]Input:[/cyan] {scenario['input']}")
        console.print("=" * 80, style="dim")

        # Create context
        context = StateContext(user_input=scenario['input'])

        # Execute state machine
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Executing state machine...", total=None)
            result = agent.execute_state_machine(scenario['machine'], context, trace_execution=False)

        # Display results
        agent._display_execution_result(result, agent.state_machines[scenario['machine']])

        # Show key responses
        if result.context.current_data:
            console.print(f"\n[bold green]🤖 Agent Responses:[/bold green]")
            for key, value in result.context.current_data.items():
                if key.endswith('_response'):
                    state_name = key.replace('_response', '').replace('_', ' ').title()
                    console.print(f"\n[blue]{state_name}:[/blue]")
                    console.print(Panel(value[:200] + "..." if len(value) > 200 else value, border_style="blue"))

    # Visualize state machines
    console.print("\n🎨 State Machine Visualizations:", style="bold magenta")
    for machine_name in agent.state_machines.keys():
        console.print(f"\n--- {machine_name.replace('_', ' ').title()} ---")
        agent.visualize_state_machine(machine_name)

    # Interactive mode
    console.print("\n🎯 Interactive Mode (type 'quit' to exit)", style="bold magenta")
    console.print("Available commands:", style="dim")
    console.print("  - 'list' - List available state machines", style="dim")
    console.print("  - 'info <machine>' - Get info about a state machine", style="dim")
    console.print("  - 'run <machine> <input>' - Execute a state machine", style="dim")
    console.print("-" * 80, style="dim")

    while True:
        try:
            user_input = console.input("\n[bold cyan]Command:[/bold cyan] ")
            if user_input.lower() in ['quit', 'exit']:
                break

            parts = user_input.split(' ', 2)
            command = parts[0].lower()

            if command == 'list':
                agent.list_state_machines()
            elif command == 'info' and len(parts) > 1:
                machine_name = parts[1]
                info = agent.get_state_machine_info(machine_name)
                if info:
                    console.print(f"\n[bold blue]State Machine: {info['name']}[/bold blue]")
                    console.print(f"Description: {info['description']}")
                    console.print(f"States: {len(info['states'])}")
                    console.print(f"Transitions: {len(info['transitions'])}")
                else:
                    console.print(f"State machine '{machine_name}' not found", style="red")
            elif command == 'run' and len(parts) > 2:
                machine_name = parts[1]
                input_text = parts[2]

                if machine_name in agent.state_machines:
                    context = StateContext(user_input=input_text)
                    result = agent.execute_state_machine(machine_name, context)

                    # Show key response
                    if result.context.current_data:
                        for key, value in result.context.current_data.items():
                            if key.endswith('_response'):
                                console.print(f"\n[green]Response:[/green] {value[:300]}...")
                                break
                else:
                    console.print(f"State machine '{machine_name}' not found", style="red")
            else:
                console.print("Invalid command. Use 'list', 'info <machine>', or 'run <machine> <input>'", style="yellow")

        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
