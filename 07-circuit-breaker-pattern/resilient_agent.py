"""Resilient Agent Pattern Implementation using Circuit Breakers."""

import os
import sys
import time
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Add parent directory to path for shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utils import create_llm, validate_environment
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CallResult, CircuitState,
    FallbackStrategy, SimpleFallback, CachedFallback, DegradedServiceFallback
)

load_dotenv()

# Initialize Rich console for better output
console = Console()


@dataclass
class AgentResponse:
    """Response from resilient agent."""
    content: str
    success: bool
    used_fallback: bool = False
    circuit_state: CircuitState = CircuitState.CLOSED
    response_time: float = 0.0
    quality_score: Optional[float] = None
    error_message: str = ""


class ResilientAgentState(TypedDict):
    """State of the resilient agent graph."""
    user_input: str
    response: str
    success: bool
    used_fallback: bool
    circuit_state: CircuitState
    response_time: float
    quality_score: Optional[float]
    error_message: str


class ResilientAgent:
    """An agent that uses circuit breakers for resilient operation."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the resilient agent."""
        self.model = create_llm(model_name=model_name, temperature=0.1)
        
        # Create circuit breakers for different services
        self.circuit_breakers = self._create_circuit_breakers()
        
        # Create LangGraph workflow
        self.graph = self._create_graph()
    
    def _create_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Create circuit breakers for different services."""
        circuit_breakers = {}
        
        # Main LLM circuit breaker
        llm_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout_duration=20.0,
            quality_threshold=0.6,
            rate_limit_per_minute=30
        )
        
        # Cached fallback for LLM
        cached_fallback = CachedFallback()
        cached_fallback.add_to_cache("greeting", "Hello! I'm currently experiencing some issues, but I'm here to help.")
        cached_fallback.add_to_cache("error", "I apologize, but I'm having technical difficulties. Please try again later.")
        cached_fallback.add_to_cache("default", "I'm temporarily unable to process your request. Please try again in a few moments.")
        
        circuit_breakers["llm"] = CircuitBreaker("LLM Service", llm_config, cached_fallback)
        
        # Quality assessment circuit breaker
        quality_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=3,
            timeout_duration=10.0,
            rate_limit_per_minute=60
        )
        
        quality_fallback = SimpleFallback(0.5)  # Default quality score
        circuit_breakers["quality"] = CircuitBreaker("Quality Service", quality_config, quality_fallback)
        
        # External API circuit breaker (simulated)
        api_config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=45.0,
            success_threshold=1,
            timeout_duration=15.0,
            rate_limit_per_minute=20
        )
        
        api_fallback = SimpleFallback({"status": "unavailable", "data": "Service temporarily unavailable"})
        circuit_breakers["external_api"] = CircuitBreaker("External API", api_config, api_fallback)
        
        return circuit_breakers
    
    def _create_graph(self) -> Any:
        """
        Create the LangGraph workflow for resilient processing.
        
        graph TD
            A[Start] --> B(Process Request)
            B --> C{Should Assess Quality?}
            C -- Assess --> D(Assess Quality)
            C -- Fallback --> F(Handle Fallback)
            C -- Complete --> G[END]
            D --> E{Should Use Fallback?}
            E -- Fallback --> F
            E -- Complete --> G
            F --> G
        """
        workflow = StateGraph(ResilientAgentState)
        
        # Add nodes
        workflow.add_node("process_request", self.process_request)
        workflow.add_node("assess_quality", self.assess_quality)
        workflow.add_node("handle_fallback", self.handle_fallback)
        
        # Set entry point
        workflow.set_entry_point("process_request")
        
        # Add edges
        workflow.add_conditional_edges(
            "process_request",
            self.should_assess_quality,
            {
                "assess": "assess_quality",
                "fallback": "handle_fallback",
                "complete": END,
            },
        )
        
        workflow.add_conditional_edges(
            "assess_quality",
            self.should_use_fallback,
            {
                "fallback": "handle_fallback",
                "complete": END,
            },
        )
        
        workflow.add_edge("handle_fallback", END)
        
        return workflow.compile()
    
    def process_request(self, state: ResilientAgentState) -> Dict[str, Any]:
        """Process user request with circuit breaker protection."""
        user_input = state["user_input"]
        
        # Create LLM call function
        def llm_call():
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."),
                ("human", "{user_input}")
            ])
            
            response = self.model.invoke(
                prompt.format_messages(user_input=user_input)
            )
            return response.content
        
        # Quality evaluator function
        def evaluate_quality(response: str) -> float:
            """Evaluate response quality (simplified)."""
            # Simple heuristics for quality assessment
            if len(response) < 10:
                return 0.2
            elif len(response) < 50:
                return 0.6
            elif "I don't know" in response.lower() or "I'm not sure" in response.lower():
                return 0.4
            else:
                return 0.8
        
        # Execute with circuit breaker
        result = self.circuit_breakers["llm"].call(
            llm_call,
            quality_evaluator=evaluate_quality
        )
        
        return {
            "response": result.response if result.success else "",
            "success": result.success,
            "used_fallback": not result.success,
            "circuit_state": self.circuit_breakers["llm"].state,
            "response_time": result.duration,
            "quality_score": result.quality_score,
            "error_message": str(result.error) if result.error else ""
        }
    
    def assess_quality(self, state: ResilientAgentState) -> Dict[str, Any]:
        """Assess response quality with circuit breaker protection."""
        response = state["response"]
        
        def quality_assessment():
            """Simulate quality assessment service."""
            # Simulate some processing time
            time.sleep(0.1)
            
            # Simulate occasional failures
            if random.random() < 0.1:  # 10% failure rate
                raise Exception("Quality assessment service error")
            
            # Simple quality scoring
            score = 0.5
            if len(response) > 100:
                score += 0.2
            if "?" in response:
                score += 0.1
            if any(word in response.lower() for word in ["help", "assist", "support"]):
                score += 0.2
            
            return min(score, 1.0)
        
        result = self.circuit_breakers["quality"].call(quality_assessment)
        
        return {
            "quality_score": result.response if result.success else 0.5
        }
    
    def handle_fallback(self, state: ResilientAgentState) -> Dict[str, Any]:
        """Handle fallback scenarios."""
        # Determine appropriate fallback response
        if state.get("error_message"):
            if "timeout" in state["error_message"].lower():
                fallback_response = "I'm taking longer than usual to respond. Here's a quick answer: I'm here to help with your questions."
            elif "rate limit" in state["error_message"].lower():
                fallback_response = "I'm currently handling many requests. Please wait a moment and try again."
            else:
                fallback_response = "I'm experiencing some technical difficulties, but I'm still here to assist you."
        else:
            fallback_response = "I'm operating in a limited capacity right now, but I'll do my best to help."
        
        return {
            "response": fallback_response,
            "success": True,
            "used_fallback": True
        }
    
    def should_assess_quality(self, state: ResilientAgentState) -> str:
        """Determine if quality assessment is needed."""
        if not state["success"]:
            return "fallback"
        elif state["response"] and len(state["response"]) > 0:
            return "assess"
        else:
            return "complete"
    
    def should_use_fallback(self, state: ResilientAgentState) -> str:
        """Determine if fallback should be used based on quality."""
        quality_score = state.get("quality_score")
        
        if quality_score is not None and quality_score < 0.3:
            return "fallback"
        else:
            return "complete"
    
    def query(self, user_input: str) -> AgentResponse:
        """Query the resilient agent."""
        try:
            initial_state = {
                "user_input": user_input,
                "response": "",
                "success": False,
                "used_fallback": False,
                "circuit_state": CircuitState.CLOSED,
                "response_time": 0.0,
                "quality_score": None,
                "error_message": ""
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return AgentResponse(
                content=final_state["response"],
                success=final_state["success"],
                used_fallback=final_state["used_fallback"],
                circuit_state=final_state["circuit_state"],
                response_time=final_state["response_time"],
                quality_score=final_state.get("quality_score"),
                error_message=final_state.get("error_message", "")
            )
            
        except Exception as e:
            return AgentResponse(
                content=f"Critical error: {str(e)}",
                success=False,
                used_fallback=True,
                error_message=str(e)
            )
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        status = {}
        
        for name, cb in self.circuit_breakers.items():
            metrics = cb.get_metrics()
            status[name] = {
                "state": metrics.current_state.value,
                "success_rate": f"{metrics.success_rate:.2%}",
                "total_calls": metrics.total_calls,
                "consecutive_failures": metrics.consecutive_failures,
                "consecutive_successes": metrics.consecutive_successes,
                "avg_response_time": f"{metrics.average_response_time:.3f}s"
            }
        
        return status
    
    def display_circuit_breaker_status(self):
        """Display circuit breaker status in a formatted table."""
        console.print("\n🔧 Circuit Breaker Status", style="bold blue")
        
        status_table = Table(show_header=True, header_style="bold magenta")
        status_table.add_column("Service", style="cyan")
        status_table.add_column("State", style="green")
        status_table.add_column("Success Rate", style="yellow")
        status_table.add_column("Total Calls", style="blue")
        status_table.add_column("Failures", style="red")
        status_table.add_column("Avg Response", style="white")
        
        for name, cb in self.circuit_breakers.items():
            metrics = cb.get_metrics()
            
            # Color code the state
            state_color = "green" if metrics.current_state == CircuitState.CLOSED else \
                         "yellow" if metrics.current_state == CircuitState.HALF_OPEN else "red"
            
            status_table.add_row(
                name.replace("_", " ").title(),
                f"[{state_color}]{metrics.current_state.value.upper()}[/{state_color}]",
                f"{metrics.success_rate:.1%}",
                str(metrics.total_calls),
                str(metrics.consecutive_failures),
                f"{metrics.average_response_time:.3f}s"
            )
        
        console.print(status_table)
    
    def simulate_service_degradation(self, service_name: str):
        """Simulate service degradation for testing."""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name].force_open()
            console.print(f"🔴 Simulated failure for {service_name}", style="red")
    
    def reset_circuit_breaker(self, service_name: str):
        """Reset a specific circuit breaker."""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name].reset()
            console.print(f"🔄 Reset circuit breaker for {service_name}", style="green")
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()
        console.print("🔄 Reset all circuit breakers", style="green")


def simulate_unreliable_service():
    """Simulate an unreliable external service for testing."""
    # Randomly fail to simulate real-world conditions
    failure_rate = 0.3  # 30% failure rate

    if random.random() < failure_rate:
        # Simulate different types of failures
        failure_type = random.choice(["timeout", "error", "rate_limit"])

        if failure_type == "timeout":
            time.sleep(2)  # Simulate slow response
            raise TimeoutError("Service timeout")
        elif failure_type == "error":
            raise Exception("Service internal error")
        else:
            raise Exception("Rate limit exceeded")

    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.5))
    return "Service response: Operation completed successfully"


def main():
    """Demo the resilient agent with circuit breakers."""
    console.print(Panel.fit("🛡️ Circuit Breaker Pattern Demo", style="bold blue"))

    # Validate environment
    if not validate_environment():
        console.print("❌ Environment validation failed", style="bold red")
        return

    # Initialize agent
    console.print("Initializing resilient agent with circuit breakers...", style="yellow")
    agent = ResilientAgent()
    console.print("✅ Agent initialized successfully!", style="green")

    # Display initial circuit breaker status
    console.print("\n📊 Initial Circuit Breaker Status:", style="bold blue")
    agent.display_circuit_breaker_status()

    # Demo scenarios
    demo_scenarios = [
        {
            "title": "Normal Operation",
            "queries": [
                "What is artificial intelligence?",
                "Explain machine learning in simple terms",
                "How do neural networks work?"
            ],
            "description": "Test normal operation with circuit breakers closed"
        },
        {
            "title": "Service Degradation Simulation",
            "queries": [
                "What are the benefits of cloud computing?",
                "Explain database normalization",
                "What is software architecture?"
            ],
            "description": "Simulate service failures and observe circuit breaker behavior",
            "simulate_failure": True
        },
        {
            "title": "Recovery Testing",
            "queries": [
                "What is DevOps?",
                "Explain microservices architecture",
                "What are design patterns?"
            ],
            "description": "Test recovery after circuit breaker reset"
        }
    ]

    console.print("\n🚀 Running Circuit Breaker Demonstrations", style="bold cyan")
    console.print("Each demo shows how circuit breakers protect against failures and enable graceful degradation.\n")

    for i, scenario in enumerate(demo_scenarios, 1):
        console.print(f"[bold]Demo {i}: {scenario['title']}[/bold]")
        console.print(f"[dim]Description: {scenario['description']}[/dim]")
        console.print("-" * 80, style="dim")

        # Simulate failure if requested
        if scenario.get("simulate_failure"):
            console.print("🔴 Simulating service degradation...", style="red")
            agent.simulate_service_degradation("llm")
            time.sleep(1)

        # Process queries
        for j, query in enumerate(scenario["queries"], 1):
            console.print(f"\n[cyan]Query {j}:[/cyan] {query}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Processing with circuit breaker protection...", total=None)
                response = agent.query(query)

            # Display response
            console.print(f"\n[green]🤖 Response:[/green]")

            # Color code based on success and fallback usage
            if response.success and not response.used_fallback:
                border_style = "green"
                title = "✅ Normal Response"
            elif response.success and response.used_fallback:
                border_style = "yellow"
                title = "⚠️ Fallback Response"
            else:
                border_style = "red"
                title = "❌ Failed Response"

            console.print(Panel(response.content, border_style=border_style, title=title))

            # Display response metrics
            console.print(f"\n[bold blue]📊 Response Metrics:[/bold blue]")

            metrics_table = Table(show_header=True, header_style="bold magenta")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")

            metrics_table.add_row("Success", "✅ Yes" if response.success else "❌ No")
            metrics_table.add_row("Used Fallback", "⚠️ Yes" if response.used_fallback else "✅ No")
            metrics_table.add_row("Circuit State", response.circuit_state.value.upper())
            metrics_table.add_row("Response Time", f"{response.response_time:.3f}s")

            if response.quality_score is not None:
                quality_color = "green" if response.quality_score >= 0.7 else "yellow" if response.quality_score >= 0.5 else "red"
                metrics_table.add_row("Quality Score", f"[{quality_color}]{response.quality_score:.2f}[/{quality_color}]")

            if response.error_message:
                metrics_table.add_row("Error", response.error_message[:50] + "..." if len(response.error_message) > 50 else response.error_message)

            console.print(metrics_table)

            # Small delay between queries
            time.sleep(0.5)

        # Display circuit breaker status after scenario
        console.print(f"\n[bold blue]📊 Circuit Breaker Status After {scenario['title']}:[/bold blue]")
        agent.display_circuit_breaker_status()

        # Reset circuit breakers for recovery demo
        if scenario.get("simulate_failure") and i < len(demo_scenarios):
            console.print("\n🔄 Resetting circuit breakers for next demo...", style="cyan")
            agent.reset_all_circuit_breakers()
            time.sleep(1)

        console.print("\n" + "="*80 + "\n")

    # Demonstrate circuit breaker benefits
    console.print("[bold green]✨ Circuit Breaker Benefits Demonstrated:[/bold green]")
    benefits = [
        "🛡️ Failure Isolation: Prevents cascading failures across services",
        "⚡ Fast Failure: Quick response when services are down",
        "🔄 Automatic Recovery: Self-healing when services recover",
        "📊 Monitoring: Real-time visibility into service health",
        "🎯 Graceful Degradation: Fallback responses maintain functionality",
        "⏱️ Timeout Protection: Prevents hanging requests",
        "🚦 Rate Limiting: Protects against overload"
    ]

    for benefit in benefits:
        console.print(f"  {benefit}")

    # Interactive mode
    console.print("\n🎯 Interactive Circuit Breaker Demo (type 'quit' to exit)", style="bold magenta")
    console.print("Commands:", style="dim")
    console.print("  - 'status' - Show circuit breaker status", style="dim")
    console.print("  - 'fail <service>' - Simulate service failure", style="dim")
    console.print("  - 'reset <service>' - Reset specific circuit breaker", style="dim")
    console.print("  - 'reset all' - Reset all circuit breakers", style="dim")
    console.print("  - Any other text will be processed as a query", style="dim")
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
                else:
                    console.print(f"Unknown service: {service}. Available: {list(agent.circuit_breakers.keys())}", style="yellow")
            elif user_input.lower().startswith('reset '):
                service = user_input[6:].strip()
                if service == 'all':
                    agent.reset_all_circuit_breakers()
                elif service in agent.circuit_breakers:
                    agent.reset_circuit_breaker(service)
                else:
                    console.print(f"Unknown service: {service}. Available: {list(agent.circuit_breakers.keys())}", style="yellow")
            else:
                # Process as query
                response = agent.query(user_input)

                # Display response
                if response.success and not response.used_fallback:
                    console.print(f"\n[green]✅ Response:[/green] {response.content}")
                elif response.success and response.used_fallback:
                    console.print(f"\n[yellow]⚠️ Fallback Response:[/yellow] {response.content}")
                else:
                    console.print(f"\n[red]❌ Failed:[/red] {response.content}")

                # Show quick metrics
                console.print(f"[dim]State: {response.circuit_state.value} | Time: {response.response_time:.3f}s | Fallback: {response.used_fallback}[/dim]")

        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

    # Final status
    console.print("\n📊 Final Circuit Breaker Status:", style="bold blue")
    agent.display_circuit_breaker_status()

    console.print("\n✨ Circuit Breaker Pattern demonstration completed!", style="bold green")


if __name__ == "__main__":
    main()
