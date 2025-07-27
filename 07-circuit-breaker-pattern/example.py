#!/usr/bin/env python3
"""
Example usage of the Circuit Breaker Pattern.

This script demonstrates how to use circuit breakers to create
resilient agent systems that gracefully handle failures.
"""

import os
import time
import random
from dotenv import load_dotenv
from resilient_agent import ResilientAgent
from circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, FailureType,
    SimpleFallback, CachedFallback
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

console = Console()


def create_demo_circuit_breaker():
    """Create a demo circuit breaker for testing."""
    
    # Configuration for quick demonstration
    config = CircuitBreakerConfig(
        failure_threshold=3,      # Open after 3 failures
        recovery_timeout=5.0,     # Try recovery after 5 seconds
        success_threshold=2,      # Close after 2 successes
        timeout_duration=2.0,     # 2 second timeout
        quality_threshold=0.6,    # Minimum quality score
        rate_limit_per_minute=10  # 10 requests per minute max
    )
    
    # Cached fallback with predefined responses
    fallback = CachedFallback()
    fallback.add_to_cache("greeting", "Hello! I'm currently experiencing issues but I'm here to help.")
    fallback.add_to_cache("question", "I'm having trouble processing your question right now. Please try again later.")
    fallback.add_to_cache("default", "Service temporarily unavailable. Please try again in a few moments.")
    
    return CircuitBreaker("Demo Service", config, fallback)


def simulate_unreliable_service(failure_rate: float = 0.3):
    """Simulate an unreliable service for testing."""
    
    def service_call(request: str) -> str:
        """Simulate a service call that may fail."""
        
        # Random delay to simulate processing
        processing_time = random.uniform(0.1, 1.0)
        time.sleep(processing_time)
        
        # Simulate failures
        if random.random() < failure_rate:
            failure_types = [
                ("timeout", lambda: time.sleep(3)),  # Simulate timeout
                ("error", lambda: (_ for _ in ()).throw(Exception("Service internal error"))),
                ("quality", lambda: "Bad response")  # Low quality response
            ]
            
            failure_type, failure_action = random.choice(failure_types)
            
            if failure_type == "timeout":
                failure_action()
                return f"Processed: {request}"
            elif failure_type == "error":
                failure_action()
            else:
                return failure_action()
        
        # Successful response
        responses = [
            f"Successfully processed: {request}",
            f"Completed request: {request}",
            f"Response for: {request}",
            f"Handled: {request}"
        ]
        
        return random.choice(responses)
    
    return service_call


def quality_evaluator(response: str) -> float:
    """Evaluate response quality."""
    if not response or len(response) < 10:
        return 0.2
    elif "error" in response.lower() or "bad" in response.lower():
        return 0.3
    elif "successfully" in response.lower() or "completed" in response.lower():
        return 0.9
    else:
        return 0.7


def main():
    """Run example demonstrations of the circuit breaker pattern."""
    
    console.print(Panel.fit("Circuit Breaker Pattern Example", style="bold blue"))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ Please set OPENAI_API_KEY in your .env file", style="bold red")
        console.print("Copy .env.example to .env and add your OpenAI API key")
        return
    
    # Initialize the resilient agent
    console.print("Initializing resilient agent with circuit breakers...", style="yellow")
    agent = ResilientAgent()
    console.print("✅ Agent ready!", style="green")
    
    # Example 1: Basic Circuit Breaker Demonstration
    console.print("\n🔧 Example 1: Basic Circuit Breaker Mechanics", style="bold cyan")
    console.print("Demonstrating how circuit breakers protect against failures")
    console.print("-" * 80, style="dim")
    
    # Create demo circuit breaker
    demo_cb = create_demo_circuit_breaker()
    unreliable_service = simulate_unreliable_service(failure_rate=0.5)  # 50% failure rate
    
    # Test requests
    test_requests = [
        "Process user data",
        "Generate report", 
        "Send notification",
        "Update database",
        "Validate input",
        "Calculate metrics",
        "Create backup",
        "Send email"
    ]
    
    console.print(f"\n[green]Testing circuit breaker with {len(test_requests)} requests:[/green]")
    
    for i, request in enumerate(test_requests, 1):
        console.print(f"\n[cyan]Request {i}:[/cyan] {request}")
        
        # Make the call through circuit breaker
        result = demo_cb.call(
            unreliable_service,
            request,
            quality_evaluator=quality_evaluator
        )
        
        # Display result
        if result.success:
            status_color = "green"
            status_icon = "✅"
        else:
            status_color = "red" 
            status_icon = "❌"
        
        console.print(f"[{status_color}]{status_icon} Result:[/{status_color}] {result.response}")
        
        # Show circuit state
        state_color = "green" if demo_cb.state == CircuitState.CLOSED else \
                     "yellow" if demo_cb.state == CircuitState.HALF_OPEN else "red"
        
        console.print(f"[{state_color}]Circuit State: {demo_cb.state.value.upper()}[/{state_color}] | "
                     f"Duration: {result.duration:.3f}s | "
                     f"Failures: {demo_cb.consecutive_failures}")
        
        if result.failure_type:
            console.print(f"[red]Failure Type: {result.failure_type.value}[/red]")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Show final metrics
    console.print(f"\n[bold blue]📊 Circuit Breaker Metrics:[/bold blue]")
    metrics = demo_cb.get_metrics()
    
    metrics_table = Table(show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_table.add_row("Total Calls", str(metrics.total_calls))
    metrics_table.add_row("Successful Calls", str(metrics.successful_calls))
    metrics_table.add_row("Failed Calls", str(metrics.failed_calls))
    metrics_table.add_row("Success Rate", f"{metrics.success_rate:.1%}")
    metrics_table.add_row("Avg Response Time", f"{metrics.average_response_time:.3f}s")
    metrics_table.add_row("Current State", metrics.current_state.value.upper())
    
    console.print(metrics_table)
    
    # Example 2: Resilient Agent Demonstration
    console.print("\n\n🛡️ Example 2: Resilient Agent with Multiple Circuit Breakers", style="bold cyan")
    console.print("Demonstrating how the agent handles various failure scenarios")
    console.print("-" * 80, style="dim")
    
    # Display initial status
    console.print("\n[green]Initial Circuit Breaker Status:[/green]")
    agent.display_circuit_breaker_status()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does deep learning work?",
        "What are the benefits of AI?",
        "Describe natural language processing"
    ]
    
    console.print(f"\n[green]Testing resilient agent with {len(test_queries)} queries:[/green]")
    
    for i, query in enumerate(test_queries, 1):
        console.print(f"\n[cyan]Query {i}:[/cyan] {query}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Processing with circuit breaker protection...", total=None)
            response = agent.query(query)
        
        # Display response
        if response.success and not response.used_fallback:
            console.print(f"[green]✅ Normal Response:[/green]")
            console.print(Panel(response.content[:200] + "..." if len(response.content) > 200 else response.content, 
                               border_style="green"))
        elif response.success and response.used_fallback:
            console.print(f"[yellow]⚠️ Fallback Response:[/yellow]")
            console.print(Panel(response.content, border_style="yellow"))
        else:
            console.print(f"[red]❌ Failed Response:[/red]")
            console.print(Panel(response.content, border_style="red"))
        
        # Show metrics
        console.print(f"[dim]Circuit: {response.circuit_state.value} | "
                     f"Time: {response.response_time:.3f}s | "
                     f"Fallback: {response.used_fallback}[/dim]")
        
        # Simulate some failures for demonstration
        if i == 3:
            console.print("\n[red]🔴 Simulating service degradation...[/red]")
            agent.simulate_service_degradation("llm")
    
    # Show final status
    console.print("\n[green]Final Circuit Breaker Status:[/green]")
    agent.display_circuit_breaker_status()
    
    # Example 3: Recovery Demonstration
    console.print("\n\n🔄 Example 3: Circuit Breaker Recovery", style="bold cyan")
    console.print("Demonstrating automatic recovery and manual reset")
    console.print("-" * 80, style="dim")
    
    console.print("\n[yellow]Resetting circuit breakers...[/yellow]")
    agent.reset_all_circuit_breakers()
    
    console.print("\n[green]Circuit breakers reset. Testing recovery:[/green]")
    
    recovery_queries = [
        "What is artificial intelligence?",
        "How do computers work?",
        "Explain software engineering"
    ]
    
    for query in recovery_queries:
        response = agent.query(query)
        
        if response.success and not response.used_fallback:
            console.print(f"[green]✅ Service recovered:[/green] {query}")
        else:
            console.print(f"[yellow]⚠️ Still using fallback:[/yellow] {query}")
    
    # Interactive demonstration
    console.print("\n🎯 Interactive Circuit Breaker Demo", style="bold magenta")
    console.print("Try your own queries and see how circuit breakers protect the system!")
    console.print("Commands: 'status', 'fail <service>', 'reset <service>', or ask any question")
    console.print("Type 'quit' to exit")
    console.print("-" * 80, style="dim")
    
    while True:
        try:
            user_input = console.input("\n[bold cyan]Input:[/bold cyan] ")
            
            if user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'status':
                agent.display_circuit_breaker_status()
            elif user_input.lower().startswith('fail '):
                service = user_input[5:].strip()
                if service in agent.circuit_breakers:
                    agent.simulate_service_degradation(service)
                    console.print(f"[red]🔴 Simulated failure for {service}[/red]")
                else:
                    console.print(f"[yellow]Unknown service: {service}[/yellow]")
                    console.print(f"Available services: {list(agent.circuit_breakers.keys())}")
            elif user_input.lower().startswith('reset '):
                service = user_input[6:].strip()
                if service == 'all':
                    agent.reset_all_circuit_breakers()
                    console.print("[green]🔄 Reset all circuit breakers[/green]")
                elif service in agent.circuit_breakers:
                    agent.reset_circuit_breaker(service)
                    console.print(f"[green]🔄 Reset {service} circuit breaker[/green]")
                else:
                    console.print(f"[yellow]Unknown service: {service}[/yellow]")
            else:
                # Process as query
                response = agent.query(user_input)
                
                if response.success and not response.used_fallback:
                    console.print(f"\n[green]✅ Response:[/green] {response.content[:150]}...")
                elif response.success and response.used_fallback:
                    console.print(f"\n[yellow]⚠️ Fallback:[/yellow] {response.content}")
                else:
                    console.print(f"\n[red]❌ Failed:[/red] {response.content}")
                
                console.print(f"[dim]State: {response.circuit_state.value} | Time: {response.response_time:.3f}s[/dim]")
                
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
    
    console.print("\n✨ Circuit Breaker Pattern demonstration completed!", style="bold green")
    console.print("\nKey benefits of Circuit Breaker Patterns:")
    
    benefits = [
        "🛡️ Failure isolation prevents cascading failures",
        "⚡ Fast failure detection and response",
        "🔄 Automatic recovery when services heal",
        "📊 Real-time monitoring and metrics",
        "🎯 Graceful degradation with fallback responses",
        "⏱️ Timeout protection prevents hanging requests",
        "🚦 Rate limiting protects against overload",
        "🔧 Manual control for testing and maintenance"
    ]
    
    for benefit in benefits:
        console.print(f"  {benefit}")
    
    console.print("\nTo run the full interactive demo, use: python resilient_agent.py")


if __name__ == "__main__":
    main()
