"""RAG-Enhanced Agent Pattern Implementation using LangGraph."""

import os
import sys
from typing import List, Dict, Any, Optional
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
from rich.progress import Progress, SpinnerColumn, TextColumn

from document_processor import DocumentProcessor, create_sample_documents
from vector_store import VectorStoreManager, VectorStoreConfig, SearchResult

load_dotenv()

# Initialize Rich console for better output
console = Console()


@dataclass
class RAGResponse:
    """Response from RAG-enhanced agent."""
    answer: str
    sources: List[SearchResult]
    context_used: str
    confidence_score: float
    query: str


class RAGState(TypedDict):
    """State of the RAG agent graph."""
    query: str
    context: str
    sources: List[SearchResult]
    answer: str
    confidence_score: float
    needs_clarification: bool


class RAGEnhancedAgent:
    """A LangGraph-based agent enhanced with RAG capabilities."""
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        vector_store_config: Optional[VectorStoreConfig] = None
    ):
        """Initialize the RAG-enhanced agent."""
        self.model = create_llm(model_name=model_name, temperature=0.1)
        self.vector_store_config = vector_store_config or VectorStoreConfig()
        self.vector_store = VectorStoreManager(self.vector_store_config)
        self.document_processor = DocumentProcessor(
            chunk_size=self.vector_store_config.chunk_size,
            chunk_overlap=self.vector_store_config.chunk_overlap
        )
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for RAG-enhanced responses."""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("evaluate_response", self.evaluate_response)
        
        # Set entry point
        workflow.set_entry_point("retrieve_context")
        
        # Add edges
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_conditional_edges(
            "generate_response",
            self.should_evaluate,
            {
                "evaluate": "evaluate_response",
                "complete": END,
            },
        )
        workflow.add_edge("evaluate_response", END)
        
        return workflow.compile()
    
    def retrieve_context(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve relevant context from the vector store."""
        query = state["query"]
        
        # Get relevant context
        search_results = self.vector_store.similarity_search(
            query=query,
            k=5,
            filter_metadata=None
        )
        
        # Format context
        context = self.vector_store.get_relevant_context(
            query=query,
            max_chunks=5,
            min_similarity=self.vector_store_config.similarity_threshold
        )
        
        return {
            "context": context,
            "sources": search_results
        }
    
    def generate_response(self, state: RAGState) -> Dict[str, Any]:
        """Generate response using retrieved context."""
        query = state["query"]
        context = state["context"]
        
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to a knowledge base. 
            Use the provided context to answer the user's question accurately and comprehensively.

Guidelines:
1. Base your answer primarily on the provided context
2. If the context doesn't contain enough information, clearly state this
3. Cite specific sources when possible
4. Be precise and avoid speculation
5. If the question cannot be answered with the given context, say so clearly

Context:
{context}"""),
            ("human", "{query}")
        ])
        
        try:
            response = self.model.invoke(
                generation_prompt.format_messages(
                    context=context,
                    query=query
                )
            )
            
            return {"answer": response.content}
            
        except Exception as e:
            return {"answer": f"Error generating response: {str(e)}"}
    
    def evaluate_response(self, state: RAGState) -> Dict[str, Any]:
        """Evaluate the quality and confidence of the response."""
        query = state["query"]
        answer = state["answer"]
        context = state["context"]
        
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI response evaluator. Assess the quality of the given answer based on the query and available context.

Provide a confidence score from 0.0 to 1.0 based on:
1. How well the answer addresses the query
2. How well the answer is supported by the context
3. The completeness and accuracy of the response
4. Whether the answer acknowledges limitations appropriately

Return only a number between 0.0 and 1.0."""),
            ("human", """Query: {query}

Context: {context}

Answer: {answer}

Confidence score:""")
        ])
        
        try:
            response = self.model.invoke(
                evaluation_prompt.format_messages(
                    query=query,
                    context=context,
                    answer=answer
                )
            )
            
            # Extract confidence score
            try:
                confidence_score = float(response.content.strip())
                confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0,1]
            except ValueError:
                confidence_score = 0.5  # Default if parsing fails
            
            return {
                "confidence_score": confidence_score,
                "needs_clarification": confidence_score < 0.6
            }
            
        except Exception as e:
            return {
                "confidence_score": 0.0,
                "needs_clarification": True
            }
    
    def should_evaluate(self, state: RAGState) -> str:
        """Determine if response should be evaluated."""
        # Always evaluate for now
        return "evaluate"
    
    def query(self, question: str) -> RAGResponse:
        """Query the RAG-enhanced agent."""
        try:
            initial_state = {
                "query": question,
                "context": "",
                "sources": [],
                "answer": "",
                "confidence_score": 0.0,
                "needs_clarification": False
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return RAGResponse(
                answer=final_state["answer"],
                sources=final_state["sources"],
                context_used=final_state["context"],
                confidence_score=final_state["confidence_score"],
                query=question
            )
            
        except Exception as e:
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                context_used="",
                confidence_score=0.0,
                query=question
            )
    
    def add_documents_from_directory(self, directory_path: str, recursive: bool = True):
        """Add documents from a directory to the knowledge base."""
        console.print(f"📁 Processing documents from: {directory_path}", style="cyan")
        
        # Process documents
        chunks, metadata = self.document_processor.process_directory(
            directory_path, recursive=recursive
        )
        
        if chunks:
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            # Display processing stats
            stats = self.document_processor.get_processing_stats(chunks)
            console.print(f"✅ Added {stats['total_chunks']} chunks from {stats['unique_sources']} sources", style="green")
        else:
            console.print("No documents found to process", style="yellow")
    
    def add_documents_from_files(self, file_paths: List[str]):
        """Add specific files to the knowledge base."""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks, metadata = self.document_processor.process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                console.print(f"❌ Error processing {file_path}: {str(e)}", style="red")
        
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
    
    def add_text(self, text: str, source: str = "direct_input"):
        """Add raw text to the knowledge base."""
        chunks = self.document_processor.process_text(text, source)
        if chunks:
            self.vector_store.add_documents(chunks, show_progress=False)
            console.print(f"✅ Added text as {len(chunks)} chunks", style="green")
    
    def search_knowledge_base(self, query: str, k: int = 5):
        """Search the knowledge base and display results."""
        self.vector_store.search_and_display(query, k=k)
    
    def get_knowledge_base_stats(self):
        """Display knowledge base statistics."""
        self.vector_store.display_stats()
    
    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base."""
        self.vector_store.clear_collection()


def main():
    """Demo the RAG-enhanced agent."""
    console.print(Panel.fit("📚 RAG-Enhanced Agent Pattern Demo", style="bold blue"))
    
    # Validate environment
    if not validate_environment():
        console.print("❌ Environment validation failed", style="bold red")
        return
    
    # Initialize agent
    console.print("Initializing RAG-enhanced agent...", style="yellow")
    agent = RAGEnhancedAgent()
    console.print("✅ Agent initialized successfully!", style="green")
    
    # Create sample documents if they don't exist
    sample_docs_path = "sample_docs"
    if not os.path.exists(sample_docs_path):
        console.print("Creating sample documents for demonstration...", style="cyan")
        create_sample_documents(sample_docs_path)
    
    # Add sample documents to knowledge base
    console.print("Adding sample documents to knowledge base...", style="cyan")
    agent.add_documents_from_directory(sample_docs_path)
    
    # Display knowledge base stats
    console.print("\n📊 Knowledge Base Statistics:", style="bold blue")
    agent.get_knowledge_base_stats()
    
    # Demo queries
    demo_queries = [
        "What is artificial intelligence and what are its main applications?",
        "Explain the different types of machine learning",
        "How do RAG systems work and what are their benefits?",
        "What are the best practices for machine learning projects?",
        "What challenges exist in AI development?"
    ]
    
    console.print("\n🚀 Running demo queries...", style="bold cyan")
    
    for i, query in enumerate(demo_queries, 1):
        console.print(f"\n[bold]Query {i}:[/bold] {query}")
        console.print("-" * 80, style="dim")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Processing query...", total=None)
            response = agent.query(query)
        
        # Display response
        console.print(f"\n[bold green]Answer:[/bold green]")
        console.print(Panel(response.answer, border_style="green"))
        
        # Display confidence and sources
        confidence_percent = response.confidence_score * 100
        confidence_color = "green" if confidence_percent >= 70 else "yellow" if confidence_percent >= 50 else "red"
        console.print(f"\n[bold blue]Confidence:[/bold blue] [{confidence_color}]{confidence_percent:.1f}%[/{confidence_color}]")
        
        if response.sources:
            console.print(f"\n[bold blue]Sources ({len(response.sources)}):[/bold blue]")
            for j, source in enumerate(response.sources[:3], 1):  # Show top 3 sources
                source_name = os.path.basename(source.source)
                similarity_percent = source.similarity_score * 100
                console.print(f"  {j}. {source_name} (similarity: {similarity_percent:.1f}%)")
    
    # Interactive mode
    console.print("\n🎯 Interactive Mode (type 'quit' to exit)", style="bold magenta")
    console.print("You can ask questions about AI, machine learning, and RAG systems", style="dim")
    console.print("-" * 80, style="dim")
    
    while True:
        try:
            user_query = console.input("\n[bold cyan]Your question:[/bold cyan] ")
            if user_query.lower() in ['quit', 'exit']:
                break
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Searching knowledge base...", total=None)
                response = agent.query(user_query)
            
            console.print(f"\n[bold green]Answer:[/bold green]")
            console.print(Panel(response.answer, border_style="green"))
            
            confidence_percent = response.confidence_score * 100
            confidence_color = "green" if confidence_percent >= 70 else "yellow" if confidence_percent >= 50 else "red"
            console.print(f"\n[{confidence_color}]Confidence: {confidence_percent:.1f}%[/{confidence_color}]")
            
            if response.sources:
                console.print(f"[dim]Sources: {len(response.sources)} relevant documents found[/dim]")
                
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
