"""Tool-Using Agent Pattern Implementation using LangGraph."""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path for shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utils import create_llm, validate_environment

from tools import get_tools

load_dotenv()

# Initialize Rich console for better output
console = Console()


class GraphState(TypedDict):
    """State of the agent graph."""
    messages: Annotated[List[AnyMessage], add_messages]


class ToolUsingAgent:
    """A LangGraph-based agent that can use multiple tools to solve problems."""
    
    def __init__(self, model_name: str = None):
        """Initialize the tool-using agent."""
        self.tools = get_tools()
        self.model = create_llm(model_name=model_name, temperature=0)
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def agent_node(self, state: GraphState) -> Dict[str, Any]:
        """The main agent reasoning node."""
        messages = state["messages"]
        response = self.model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(self, state: GraphState) -> str:
        """Determine whether to continue with tool calls or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if last_message.tool_calls:
            return "continue"
        else:
            return "end"
    
    def run(self, query: str) -> str:
        """Run the agent on a given query."""
        try:
            # Create initial state
            initial_state = {"messages": [HumanMessage(content=query)]}
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Extract the final response
            final_message = result["messages"][-1]
            return final_message.content
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def stream(self, query: str):
        """Stream the agent's execution step by step."""
        try:
            initial_state = {"messages": [HumanMessage(content=query)]}
            
            for output in self.graph.stream(initial_state):
                for node, messages in output.items():
                    print(f"--- {node.upper()} ---")
                    if "messages" in messages:
                        for msg in messages["messages"]:
                            if hasattr(msg, 'content'):
                                print(f"Content: {msg.content}")
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                print(f"Tool calls: {msg.tool_calls}")
                    print()
                    
        except Exception as e:
            print(f"Error during streaming: {str(e)}")


def main():
    """Demo the tool-using agent."""
    console.print(Panel.fit("🤖 Tool-Using Agent Pattern Demo", style="bold blue"))

    # Validate environment
    if not validate_environment():
        console.print("❌ Environment validation failed", style="bold red")
        return

    # Initialize agent
    console.print("Initializing agent...", style="yellow")
    agent = ToolUsingAgent()
    console.print("✅ Agent initialized successfully!", style="green")

    # Demo queries
    demo_queries = [
        "What is 157 multiplied by 234?",
        "Search for information about LangGraph",
        "Write 'Hello, World!' to a file called hello.txt",
        "List the files in the current directory",
        "What's the square root of 144?",
    ]

    console.print("\n🚀 Running demo queries...", style="bold cyan")

    for i, query in enumerate(demo_queries, 1):
        console.print(f"\n[bold]Query {i}:[/bold] {query}")
        console.print("-" * 60, style="dim")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Processing...", total=None)
            response = agent.run(query)

        console.print(f"[bold green]Response:[/bold green] {response}")

    # Interactive mode
    console.print("\n🎯 Interactive Mode (type 'quit' to exit)", style="bold magenta")
    console.print("-" * 60, style="dim")

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            if user_input.lower() in ['quit', 'exit']:
                break

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Thinking...", total=None)
                response = agent.run(user_input)

            console.print(f"[bold green]Agent:[/bold green] {response}")

        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()