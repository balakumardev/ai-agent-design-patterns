"""Vector store management for RAG-Enhanced Agent."""

import os
import json
import uuid
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    collection_name: str = "rag_documents"
    persist_directory: str = "./chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.7


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    source: str
    chunk_index: int


class VectorStoreManager:
    """Manages vector store operations for RAG system."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize the vector store manager."""
        self.config = config
        # Initialize embeddings with HuggingFace (local, no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.persist_directory = Path(config.persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load existing vector store."""
        try:
            # Use the existing ChromaDB client to avoid conflicts
            self.vector_store = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                client=self.chroma_client
            )
            console.print(f"✅ Vector store initialized: {self.config.collection_name}", style="green")
        except Exception as e:
            console.print(f"❌ Error initializing vector store: {str(e)}", style="red")
            # Try to reset and create a new client with a unique directory
            import tempfile
            import uuid
            unique_dir = tempfile.mkdtemp(prefix=f"chroma_{uuid.uuid4().hex[:8]}_")
            console.print(f"🔄 Retrying with unique directory: {unique_dir}", style="yellow")

            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=unique_dir,
                    settings=Settings(anonymized_telemetry=False)
                )
                self.vector_store = Chroma(
                    collection_name=self.config.collection_name,
                    embedding_function=self.embeddings,
                    client=self.chroma_client
                )
                console.print(f"✅ Vector store initialized with unique directory", style="green")
            except Exception as retry_error:
                console.print(f"❌ Failed to initialize vector store: {str(retry_error)}", style="red")
                raise
    
    def add_documents(self, documents: List[Document], show_progress: bool = True) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            console.print("No documents to add", style="yellow")
            return []
        
        console.print(f"Adding {len(documents)} documents to vector store...", style="cyan")
        
        try:
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console
                ) as progress:
                    task = progress.add_task("Creating embeddings...", total=len(documents))
                    
                    # Add documents in batches to show progress
                    batch_size = 50
                    document_ids = []
                    
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        batch_ids = self.vector_store.add_documents(batch)
                        document_ids.extend(batch_ids)
                        progress.advance(task, len(batch))
            else:
                document_ids = self.vector_store.add_documents(documents)
            
            console.print(f"✅ Added {len(documents)} documents to vector store", style="green")
            return document_ids
            
        except Exception as e:
            console.print(f"❌ Error adding documents: {str(e)}", style="red")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search and return results."""
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_metadata
            )
            
            # Convert to SearchResult objects
            search_results = []
            for doc, score in results:
                search_result = SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    similarity_score=1 - score,  # Convert distance to similarity
                    source=doc.metadata.get('source_file', doc.metadata.get('source', 'unknown')),
                    chunk_index=doc.metadata.get('chunk_index', 0)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            console.print(f"❌ Error during similarity search: {str(e)}", style="red")
            return []
    
    def get_relevant_context(
        self, 
        query: str, 
        max_chunks: int = 5, 
        min_similarity: float = None
    ) -> str:
        """Get relevant context for a query as a formatted string."""
        if min_similarity is None:
            min_similarity = self.config.similarity_threshold
        
        results = self.similarity_search(query, k=max_chunks)
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result.similarity_score >= min_similarity
        ]
        
        if not filtered_results:
            return "No relevant context found."
        
        # Format context
        context_parts = []
        for i, result in enumerate(filtered_results, 1):
            source_info = f"Source: {Path(result.source).name}"
            if result.chunk_index > 0:
                source_info += f" (chunk {result.chunk_index + 1})"
            
            context_parts.append(
                f"[Context {i}] {source_info}\n"
                f"{result.content}\n"
            )
        
        return "\n".join(context_parts)
    
    def delete_documents(self, filter_metadata: Dict[str, Any]) -> int:
        """Delete documents matching the filter criteria."""
        try:
            # Get collection directly for deletion
            collection = self.chroma_client.get_collection(self.config.collection_name)
            
            # Get documents matching filter
            results = collection.get(where=filter_metadata)
            
            if not results['ids']:
                console.print("No documents found matching filter criteria", style="yellow")
                return 0
            
            # Delete the documents
            collection.delete(ids=results['ids'])
            
            deleted_count = len(results['ids'])
            console.print(f"✅ Deleted {deleted_count} documents", style="green")
            return deleted_count
            
        except Exception as e:
            console.print(f"❌ Error deleting documents: {str(e)}", style="red")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            collection = self.chroma_client.get_collection(self.config.collection_name)
            count = collection.count()
            
            # Get sample of documents to analyze
            sample_size = min(100, count)
            if count > 0:
                sample = collection.get(limit=sample_size, include=['metadatas'])
                
                # Analyze metadata
                file_types = {}
                sources = set()
                total_tokens = 0
                
                for metadata in sample['metadatas']:
                    if metadata:
                        file_type = metadata.get('file_type', 'unknown')
                        source = metadata.get('source_file', metadata.get('source', 'unknown'))
                        tokens = metadata.get('token_count', 0)
                        
                        file_types[file_type] = file_types.get(file_type, 0) + 1
                        sources.add(source)
                        total_tokens += tokens
                
                return {
                    'total_documents': count,
                    'unique_sources': len(sources),
                    'file_types': file_types,
                    'estimated_total_tokens': int(total_tokens * (count / sample_size)) if sample_size > 0 else 0,
                    'avg_tokens_per_chunk': total_tokens / sample_size if sample_size > 0 else 0
                }
            else:
                return {
                    'total_documents': 0,
                    'unique_sources': 0,
                    'file_types': {},
                    'estimated_total_tokens': 0,
                    'avg_tokens_per_chunk': 0
                }
                
        except Exception as e:
            console.print(f"❌ Error getting collection stats: {str(e)}", style="red")
            return {}
    
    def display_stats(self):
        """Display vector store statistics in a formatted table."""
        stats = self.get_collection_stats()
        
        if not stats:
            console.print("No statistics available", style="yellow")
            return
        
        # Main stats table
        stats_table = Table(title="Vector Store Statistics", show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Documents", f"{stats['total_documents']:,}")
        stats_table.add_row("Unique Sources", str(stats['unique_sources']))
        stats_table.add_row("Estimated Total Tokens", f"{stats['estimated_total_tokens']:,}")
        stats_table.add_row("Avg Tokens per Chunk", f"{stats['avg_tokens_per_chunk']:.1f}")
        
        console.print(stats_table)
        
        # File types table
        if stats['file_types']:
            file_types_table = Table(title="File Types", show_header=True, header_style="bold blue")
            file_types_table.add_column("File Type", style="cyan")
            file_types_table.add_column("Count", style="green")
            
            for file_type, count in stats['file_types'].items():
                file_types_table.add_row(file_type, str(count))
            
            console.print(file_types_table)
    
    def search_and_display(self, query: str, k: int = 5):
        """Search and display results in a formatted way."""
        console.print(f"\n🔍 Searching for: [bold cyan]{query}[/bold cyan]")
        
        results = self.similarity_search(query, k=k)
        
        if not results:
            console.print("No results found", style="yellow")
            return
        
        # Display results
        for i, result in enumerate(results, 1):
            similarity_percent = result.similarity_score * 100
            source_name = Path(result.source).name
            
            console.print(f"\n[bold]Result {i}[/bold] (Similarity: {similarity_percent:.1f}%)")
            console.print(f"[blue]Source:[/blue] {source_name}")
            
            if result.chunk_index > 0:
                console.print(f"[blue]Chunk:[/blue] {result.chunk_index + 1}")
            
            # Show content preview
            content_preview = result.content[:300] + "..." if len(result.content) > 300 else result.content
            console.print(f"[green]Content:[/green] {content_preview}")
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            collection = self.chroma_client.get_collection(self.config.collection_name)
            collection.delete()
            console.print("✅ Collection cleared", style="green")
        except Exception as e:
            console.print(f"❌ Error clearing collection: {str(e)}", style="red")
    
    def save_config(self, config_path: str = "vector_store_config.json"):
        """Save vector store configuration to file."""
        config_dict = asdict(self.config)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        console.print(f"✅ Configuration saved to {config_path}", style="green")
    
    @classmethod
    def load_config(cls, config_path: str = "vector_store_config.json") -> 'VectorStoreManager':
        """Load vector store configuration from file."""
        if not Path(config_path).exists():
            console.print(f"Config file not found: {config_path}, using defaults", style="yellow")
            return cls(VectorStoreConfig())
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = VectorStoreConfig(**config_dict)
        return cls(config)
