#!/usr/bin/env python3
"""
Example usage of the State Machine Pattern.

This script demonstrates how to create and use state machines for
predictable agent behavior with explicit states and transitions.
"""

import os
from dotenv import load_dotenv
from state_machine_agent import StateMachineAgent
from state_models import (
    StateContext, StateMachineDefinition, Transition, TransitionCondition,
    InitialState, FinalState, ErrorState, ProcessingState, DecisionState
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load environment variables
load_dotenv()

console = Console()


def create_simple_workflow_example(agent: StateMachineAgent) -> StateMachineDefinition:
    """Create a simple workflow state machine for demonstration."""
    
    # Create states
    initial = InitialState("start", "Workflow started")
    
    # Input validation state
    def validate_input(context: StateContext) -> StateContext:
        """Validate user input."""
        if len(context.user_input.strip()) < 5:
            context.error_message = "Input too short (minimum 5 characters)"
        else:
            context.current_data['validated_input'] = context.user_input.strip()
            context.add_to_history("Input validated successfully")
        return context
    
    validate = ProcessingState("validate", "Validate user input", validate_input)
    
    # Processing state
    process = agent.create_llm_processing_state(
        "process",
        "You are a helpful assistant. Process the user's input and provide a thoughtful response. Be concise but informative.",
        "Process the validated input"
    )
    
    # Quality check state
    quality_check = agent.create_llm_decision_state(
        "quality_check",
        "Review the generated response for quality. Is it helpful, accurate, and appropriate? Respond with 'good' if satisfactory or 'needs_improvement' if not.",
        ["good", "needs_improvement"],
        "Check response quality"
    )
    
    # Improvement state
    improve = agent.create_llm_processing_state(
        "improve",
        "The previous response needs improvement. Enhance it to be more helpful, accurate, and comprehensive while maintaining clarity.",
        "Improve the response"
    )
    
    # Finalization state
    def finalize_response(context: StateContext) -> StateContext:
        """Finalize the response."""
        # Get the best available response
        if 'improve_response' in context.current_data:
            final_response = context.current_data['improve_response']
        else:
            final_response = context.current_data.get('process_response', 'No response generated')
        
        context.current_data['final_response'] = final_response
        context.add_to_history("Response finalized")
        return context
    
    finalize = ProcessingState("finalize", "Finalize the response", finalize_response)
    
    final = FinalState("end", "Workflow completed")
    error = ErrorState("error", "Workflow error")
    
    # Create state machine definition
    definition = StateMachineDefinition(
        name="simple_workflow",
        description="Simple workflow with validation, processing, and quality control",
        initial_state="start"
    )
    
    # Add states
    for state in [initial, validate, process, quality_check, improve, finalize, final, error]:
        definition.add_state(state)
    
    # Add transitions
    definition.add_transition(Transition("start", "validate", TransitionCondition.ALWAYS))
    definition.add_transition(Transition("validate", "process", TransitionCondition.ON_SUCCESS))
    definition.add_transition(Transition("validate", "error", TransitionCondition.ON_FAILURE))
    definition.add_transition(Transition("process", "quality_check", TransitionCondition.ON_SUCCESS))
    definition.add_transition(Transition("process", "error", TransitionCondition.ON_FAILURE))
    
    # Quality-based routing
    definition.add_transition(Transition("quality_check", "finalize", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('quality_check_response', '').lower().find('good') != -1))
    definition.add_transition(Transition("quality_check", "improve", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('quality_check_response', '').lower().find('improvement') != -1))
    definition.add_transition(Transition("quality_check", "error", TransitionCondition.ON_FAILURE))
    
    # From improvement back to quality check (with limit)
    definition.add_transition(Transition("improve", "quality_check", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.iteration_count < 5))  # Limit improvement cycles
    definition.add_transition(Transition("improve", "finalize", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.iteration_count >= 5))  # Force finalization after 5 iterations
    definition.add_transition(Transition("improve", "error", TransitionCondition.ON_FAILURE))
    
    definition.add_transition(Transition("finalize", "end", TransitionCondition.ON_SUCCESS))
    definition.add_transition(Transition("finalize", "error", TransitionCondition.ON_FAILURE))
    
    return definition


def create_decision_tree_example(agent: StateMachineAgent) -> StateMachineDefinition:
    """Create a decision tree state machine for content categorization."""
    
    # Create states
    initial = InitialState("start", "Content categorization started")
    
    # Content type classification
    classify_type = agent.create_llm_decision_state(
        "classify_type",
        "Analyze the content and classify its primary type. Consider the main subject matter and purpose.",
        ["technical", "creative", "business", "educational"],
        "Classify content type"
    )
    
    # Technical content processing
    technical_processing = agent.create_llm_processing_state(
        "technical_processing",
        "Process this technical content. Identify key technical concepts, provide explanations, and suggest related topics.",
        "Process technical content"
    )
    
    # Creative content processing
    creative_processing = agent.create_llm_processing_state(
        "creative_processing",
        "Process this creative content. Analyze the creative elements, style, and provide constructive feedback or enhancement suggestions.",
        "Process creative content"
    )
    
    # Business content processing
    business_processing = agent.create_llm_processing_state(
        "business_processing",
        "Process this business content. Analyze business implications, identify key insights, and provide strategic recommendations.",
        "Process business content"
    )
    
    # Educational content processing
    educational_processing = agent.create_llm_processing_state(
        "educational_processing",
        "Process this educational content. Structure the information for learning, identify key concepts, and suggest learning objectives.",
        "Process educational content"
    )
    
    # Complexity assessment
    assess_complexity = agent.create_llm_decision_state(
        "assess_complexity",
        "Assess the complexity level of the processed content. Consider depth, technical difficulty, and comprehensiveness.",
        ["basic", "intermediate", "advanced"],
        "Assess content complexity"
    )
    
    # Final processing based on complexity
    basic_finalization = agent.create_llm_processing_state(
        "basic_finalization",
        "Finalize this basic-level content. Ensure it's accessible and easy to understand for beginners.",
        "Finalize basic content"
    )
    
    intermediate_finalization = agent.create_llm_processing_state(
        "intermediate_finalization",
        "Finalize this intermediate-level content. Balance detail with clarity for users with some background knowledge.",
        "Finalize intermediate content"
    )
    
    advanced_finalization = agent.create_llm_processing_state(
        "advanced_finalization",
        "Finalize this advanced-level content. Provide comprehensive detail and assume expert-level understanding.",
        "Finalize advanced content"
    )
    
    final = FinalState("end", "Content categorization completed")
    error = ErrorState("error", "Content categorization error")
    
    # Create state machine definition
    definition = StateMachineDefinition(
        name="content_categorization",
        description="Decision tree for content categorization and processing",
        initial_state="start"
    )
    
    # Add states
    for state in [initial, classify_type, technical_processing, creative_processing, 
                  business_processing, educational_processing, assess_complexity,
                  basic_finalization, intermediate_finalization, advanced_finalization, final, error]:
        definition.add_state(state)
    
    # Add transitions
    definition.add_transition(Transition("start", "classify_type", TransitionCondition.ALWAYS))
    
    # Type-based routing
    definition.add_transition(Transition("classify_type", "technical_processing", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('classify_type_response', '').lower().find('technical') != -1))
    definition.add_transition(Transition("classify_type", "creative_processing", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('classify_type_response', '').lower().find('creative') != -1))
    definition.add_transition(Transition("classify_type", "business_processing", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('classify_type_response', '').lower().find('business') != -1))
    definition.add_transition(Transition("classify_type", "educational_processing", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('classify_type_response', '').lower().find('educational') != -1))
    
    # From processing states to complexity assessment
    for processing_state in ["technical_processing", "creative_processing", "business_processing", "educational_processing"]:
        definition.add_transition(Transition(processing_state, "assess_complexity", TransitionCondition.ON_SUCCESS))
        definition.add_transition(Transition(processing_state, "error", TransitionCondition.ON_FAILURE))
    
    # Complexity-based routing
    definition.add_transition(Transition("assess_complexity", "basic_finalization", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('assess_complexity_response', '').lower().find('basic') != -1))
    definition.add_transition(Transition("assess_complexity", "intermediate_finalization", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('assess_complexity_response', '').lower().find('intermediate') != -1))
    definition.add_transition(Transition("assess_complexity", "advanced_finalization", TransitionCondition.ON_CONDITION,
                                       lambda ctx: ctx.current_data.get('assess_complexity_response', '').lower().find('advanced') != -1))
    
    # From finalization to end
    for finalization_state in ["basic_finalization", "intermediate_finalization", "advanced_finalization"]:
        definition.add_transition(Transition(finalization_state, "end", TransitionCondition.ON_SUCCESS))
        definition.add_transition(Transition(finalization_state, "error", TransitionCondition.ON_FAILURE))
    
    # Error transitions
    definition.add_transition(Transition("classify_type", "error", TransitionCondition.ON_FAILURE))
    definition.add_transition(Transition("assess_complexity", "error", TransitionCondition.ON_FAILURE))
    
    return definition


def main():
    """Run example demonstrations of the state machine pattern."""
    
    console.print(Panel.fit("State Machine Pattern Example", style="bold blue"))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ Please set OPENAI_API_KEY in your .env file", style="bold red")
        console.print("Copy .env.example to .env and add your OpenAI API key")
        return
    
    # Initialize the agent
    console.print("Initializing state machine agent...", style="yellow")
    agent = StateMachineAgent()
    console.print("✅ Agent ready!", style="green")
    
    # Create and register example state machines
    console.print("\n🏗️ Creating example state machines...", style="cyan")
    
    # Simple workflow
    simple_workflow = create_simple_workflow_example(agent)
    agent.register_state_machine(simple_workflow)
    
    # Decision tree
    decision_tree = create_decision_tree_example(agent)
    agent.register_state_machine(decision_tree)
    
    # Display registered state machines
    console.print("\n📋 Registered State Machines:", style="bold blue")
    agent.list_state_machines()
    
    # Example scenarios
    examples = [
        {
            "machine": "simple_workflow",
            "title": "Simple Workflow - Short Input",
            "input": "Hi",
            "description": "Test input validation with short input (should trigger error handling)"
        },
        {
            "machine": "simple_workflow",
            "title": "Simple Workflow - Valid Input",
            "input": "Please explain the benefits of using state machines in AI agent design",
            "description": "Test complete workflow with valid input"
        },
        {
            "machine": "content_categorization",
            "title": "Technical Content",
            "input": "Machine learning algorithms use mathematical optimization to find patterns in data. Neural networks, for example, use backpropagation to adjust weights and minimize loss functions.",
            "description": "Test decision tree with technical content"
        },
        {
            "machine": "content_categorization",
            "title": "Creative Content",
            "input": "The sunset painted the sky in brilliant shades of orange and pink, while gentle waves lapped against the shore, creating a symphony of nature's beauty.",
            "description": "Test decision tree with creative content"
        }
    ]
    
    console.print("\n🚀 Running State Machine Examples", style="bold cyan")
    console.print("Each example shows how state machines provide predictable, traceable agent behavior.\n")
    
    for i, example in enumerate(examples, 1):
        console.print(f"[bold]Example {i}: {example['title']}[/bold]")
        console.print(f"[dim]Description: {example['description']}[/dim]")
        console.print(f"[cyan]Input:[/cyan] {example['input']}")
        console.print("-" * 80, style="dim")
        
        # Create context and execute
        context = StateContext(user_input=example['input'])
        result = agent.execute_state_machine(example['machine'], context, trace_execution=False)
        
        # Display execution summary
        console.print(f"\n[green]📊 Execution Summary:[/green]")
        
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Success", "✅ Yes" if result.success else "❌ No")
        summary_table.add_row("Final State", result.final_state)
        summary_table.add_row("States Visited", str(len(result.execution_path)))
        summary_table.add_row("Execution Path", " → ".join(result.execution_path))
        summary_table.add_row("Iterations", str(result.context.iteration_count))
        
        if result.error_message:
            summary_table.add_row("Error", result.error_message)
        
        console.print(summary_table)
        
        # Show key outputs
        if result.context.current_data:
            console.print(f"\n[green]🎯 Key Outputs:[/green]")
            for key, value in result.context.current_data.items():
                if key.endswith('_response') or key in ['final_response', 'validated_input']:
                    display_key = key.replace('_response', '').replace('_', ' ').title()
                    preview = value[:150] + "..." if len(value) > 150 else value
                    console.print(f"[blue]{display_key}:[/blue] {preview}")
        
        # Show state machine benefits
        console.print(f"\n[bold green]✨ State Machine Benefits Demonstrated:[/bold green]")
        benefits = [
            f"🔄 Predictable Flow: Clear execution path through {len(result.execution_path)} states",
            f"🎯 Error Handling: {'Graceful error handling' if result.error_message else 'Successful execution'}",
            f"📊 Traceability: Complete audit trail of state transitions",
            f"🔧 Maintainability: Explicit states and transitions for easy debugging"
        ]
        
        for benefit in benefits:
            console.print(f"  {benefit}")
        
        console.print("\n" + "="*80 + "\n")
    
    # Visualize state machines
    console.print("[bold magenta]🎨 State Machine Visualizations[/bold magenta]")
    console.print("Visual representation of state machine structure:")
    console.print("-" * 80, style="dim")
    
    for machine_name in agent.state_machines.keys():
        console.print(f"\n[bold cyan]{machine_name.replace('_', ' ').title()}[/bold cyan]")
        agent.visualize_state_machine(machine_name)
    
    # Interactive demonstration
    console.print("\n[bold magenta]💬 Interactive State Machine Demo[/bold magenta]")
    console.print("Try your own inputs with the state machines!")
    console.print("Commands: 'simple <input>' or 'categorize <input>' or 'quit'")
    console.print("-" * 80, style="dim")
    
    while True:
        try:
            user_input = console.input("\n[bold cyan]Command:[/bold cyan] ")
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                console.print("Please use: 'simple <input>' or 'categorize <input>'", style="yellow")
                continue
            
            command, content = parts
            
            if command.lower() == 'simple':
                machine_name = "simple_workflow"
            elif command.lower() == 'categorize':
                machine_name = "content_categorization"
            else:
                console.print("Unknown command. Use 'simple' or 'categorize'", style="yellow")
                continue
            
            # Execute the state machine
            context = StateContext(user_input=content)
            result = agent.execute_state_machine(machine_name, context, trace_execution=False)
            
            # Show result
            console.print(f"\n[green]Result:[/green] {'Success' if result.success else 'Failed'}")
            console.print(f"[blue]Path:[/blue] {' → '.join(result.execution_path)}")
            
            if 'final_response' in result.context.current_data:
                console.print(f"\n[green]Response:[/green]")
                console.print(Panel(result.context.current_data['final_response'], border_style="green"))
            elif result.error_message:
                console.print(f"\n[red]Error:[/red] {result.error_message}")
            
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
    
    console.print("\n✨ State Machine Pattern demonstration completed!", style="bold green")
    console.print("\nKey benefits of State Machine Patterns:")
    
    benefits = [
        "🔄 Predictable behavior with explicit states and transitions",
        "🎯 Clear error handling and recovery paths",
        "📊 Complete traceability and audit trails",
        "🔧 Easy debugging and maintenance",
        "⚡ Efficient execution with defined workflows",
        "🎨 Visual representation of agent logic",
        "🔒 Controlled state transitions prevent invalid states"
    ]
    
    for benefit in benefits:
        console.print(f"  {benefit}")
    
    console.print("\nTo run the full interactive demo, use: python state_machine_agent.py")


if __name__ == "__main__":
    main()
