"""Reflection Pattern Implementation using LangGraph."""

import os
import sys
from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path for shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utils import create_llm, validate_environment

load_dotenv()

# Initialize Rich console for better output
console = Console()


@dataclass
class ReflectionResult:
    """Result of a reflection iteration."""
    content: str
    critique: str
    needs_improvement: bool
    iteration: int


class GraphState(TypedDict):
    """State of the reflection graph."""
    original_query: str
    current_response: str
    critique: str
    iteration: int
    max_iterations: int
    quality_threshold: float
    final_response: str


class ReflectionAgent:
    """A LangGraph-based agent that improves outputs through self-reflection."""
    
    def __init__(self, model_name: Optional[str] = None, max_iterations: int = 3):
        """Initialize the reflection agent."""
        self.model = create_llm(model_name=model_name, temperature=0.7)
        self.max_iterations = max_iterations
        self.graph = self._create_graph()
    
    def _create_graph(self) -> Any:
        """
        Create the LangGraph workflow for reflection.
        
        graph TD
            A[Start] --> B(Generate)
            B --> C(Reflect)
            C --> D{Should Revise?}
            D -- Yes --> E(Revise)
            D -- No --> F[END]
            E --> C
        """
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("generate", self.generate_response)
        workflow.add_node("reflect", self.reflect_on_response)
        workflow.add_node("revise", self.revise_response)
        
        # Set entry point
        workflow.set_entry_point("generate")
        
        # Add edges
        workflow.add_edge("generate", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self.should_revise,
            {
                "revise": "revise",
                "end": END,
            },
        )
        workflow.add_edge("revise", "reflect")
        
        return workflow.compile()
    
    def generate_response(self, state: GraphState) -> Dict[str, Any]:
        """Generate initial response to the query."""
        query = state["original_query"]
        
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant tasked with providing comprehensive, 
            accurate, and well-structured responses to user queries. Focus on:
            - Accuracy and completeness
            - Clear structure and organization
            - Relevant examples when appropriate
            - Proper reasoning and explanation"""),
            ("human", "{query}")
        ])
        
        response = self.model.invoke(generation_prompt.format_messages(query=query))
        
        return {
            "current_response": response.content,
            "iteration": 1
        }
    
    def reflect_on_response(self, state: GraphState) -> Dict[str, Any]:
        """Reflect on and critique the current response."""
        query = state["original_query"]
        response = state["current_response"]
        
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critical reviewer tasked with evaluating responses for quality.
            Analyze the response for:
            1. Accuracy and correctness
            2. Completeness and thoroughness
            3. Clarity and organization
            4. Relevance to the original query
            5. Areas that could be improved
            
            Provide specific, actionable feedback. If the response is already high quality,
            indicate that no further improvements are needed."""),
            ("human", """Original Query: {query}
            
            Response to Evaluate: {response}
            
            Please provide a detailed critique and suggest specific improvements.""")
        ])
        
        critique = self.model.invoke(
            reflection_prompt.format_messages(query=query, response=response)
        )
        
        return {"critique": critique.content}
    
    def revise_response(self, state: GraphState) -> Dict[str, Any]:
        """Revise the response based on the critique."""
        query = state["original_query"]
        current_response = state["current_response"]
        critique = state["critique"]
        iteration = state["iteration"]
        
        revision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are tasked with improving a response based on detailed feedback.
            Create a revised version that addresses all the points raised in the critique
            while maintaining the strengths of the original response."""),
            ("human", """Original Query: {query}
            
            Current Response: {current_response}
            
            Critique and Improvement Suggestions: {critique}
            
            Please provide an improved version that addresses the critique.""")
        ])
        
        revised_response = self.model.invoke(
            revision_prompt.format_messages(
                query=query,
                current_response=current_response,
                critique=critique
            )
        )
        
        return {
            "current_response": revised_response.content,
            "iteration": iteration + 1
        }
    
    def should_revise(self, state: GraphState) -> Literal["revise", "end"]:
        """Determine if the response should be revised or if we're done."""
        critique = state["critique"]
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        
        # Check if we've hit max iterations
        if iteration >= max_iterations:
            return "end"
        
        # Simple heuristic: if critique mentions improvements or issues, revise
        improvement_keywords = [
            "could be improved", "should include", "missing", "unclear", 
            "needs", "recommend", "suggest", "better", "enhance", "expand"
        ]
        
        if any(keyword in critique.lower() for keyword in improvement_keywords):
            return "revise"
        else:
            return "end"
    
    def run(self, query: str, max_iterations: Optional[int] = None) -> ReflectionResult:
        """Run the reflection agent on a query."""
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        try:
            initial_state = {
                "original_query": query,
                "current_response": "",
                "critique": "",
                "iteration": 0,
                "max_iterations": max_iterations,
                "quality_threshold": 0.8,
                "final_response": ""
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return ReflectionResult(
                content=final_state["current_response"],
                critique=final_state["critique"],
                needs_improvement=final_state["iteration"] < max_iterations,
                iteration=final_state["iteration"]
            )
            
        except Exception as e:
            return ReflectionResult(
                content=f"Error processing query: {str(e)}",
                critique="",
                needs_improvement=False,
                iteration=0
            )
    
    def stream_reflection(self, query: str, max_iterations: Optional[int] = None):
        """Stream the reflection process step by step."""
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        try:
            initial_state = {
                "original_query": query,
                "current_response": "",
                "critique": "",
                "iteration": 0,
                "max_iterations": max_iterations,
                "quality_threshold": 0.8,
                "final_response": ""
            }
            
            print(f"🎯 Original Query: {query}")
            print("=" * 60)
            
            for i, output in enumerate(self.graph.stream(initial_state)):
                for node_name, node_output in output.items():
                    print(f"\n📝 Step {i+1}: {node_name.upper()}")
                    print("-" * 40)
                    
                    if node_name == "generate":
                        print(f"Generated Response:\n{node_output.get('current_response', 'N/A')}")
                    elif node_name == "reflect":
                        print(f"Critique:\n{node_output.get('critique', 'N/A')}")
                    elif node_name == "revise":
                        print(f"Revised Response (Iteration {node_output.get('iteration', 'N/A')}):")
                        print(node_output.get('current_response', 'N/A'))
                    
        except Exception as e:
            print(f"Error during streaming: {str(e)}")


def main():
    """Demo the reflection agent."""
    console.print(Panel.fit("🤖 Reflection Pattern Demo", style="bold blue"))

    # Validate environment
    if not validate_environment():
        console.print("❌ Environment validation failed", style="bold red")
        return

    # Initialize agent
    console.print("Initializing reflection agent...", style="yellow")
    agent = ReflectionAgent(max_iterations=3)
    console.print("✅ Agent initialized successfully!", style="green")

    # Demo queries that benefit from reflection
    demo_queries = [
        "Explain the concept of machine learning to a 12-year-old",
        "Write a short story about a robot discovering emotions",
        "Compare the pros and cons of renewable energy sources",
        "Explain why Python is popular for data science"
    ]

    console.print("\n🚀 Running demo queries with reflection...", style="bold cyan")

    for i, query in enumerate(demo_queries, 1):
        console.print(f"\n[bold]🎯 Query {i}:[/bold] {query}")
        console.print("=" * 80, style="dim")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Reflecting and improving...", total=None)
            result = agent.run(query, max_iterations=2)

        console.print(f"\n[bold green]📊 Final Response (after {result.iteration} iterations):[/bold green]")
        console.print(Panel(result.content, border_style="green"))

        if result.critique:
            console.print(f"\n[bold yellow]🔍 Final Critique:[/bold yellow]")
            console.print(Panel(result.critique, border_style="yellow"))

    # Interactive mode
    console.print("\n🎯 Interactive Mode (type 'quit' to exit)", style="bold magenta")
    console.print("📝 Use 'stream' prefix to see the reflection process step by step", style="dim")
    console.print("-" * 80, style="dim")

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            if user_input.lower() in ['quit', 'exit']:
                break

            if user_input.lower().startswith('stream '):
                query = user_input[7:]  # Remove 'stream ' prefix
                console.print("\n🔄 Streaming Reflection Process...", style="bold yellow")
                agent.stream_reflection(query, max_iterations=2)
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Reflecting...", total=None)
                    result = agent.run(user_input, max_iterations=2)

                console.print(f"\n[bold green]Agent (with reflection):[/bold green]")
                console.print(Panel(result.content, border_style="green"))
                console.print(f"[dim]Reflection iterations: {result.iteration}[/dim]")

        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()