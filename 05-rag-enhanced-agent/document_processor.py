"""Document processing utilities for RAG-Enhanced Agent."""

import os
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import tiktoken
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""
    file_path: str
    file_type: str
    file_size: int
    chunk_count: int
    total_tokens: int
    processed_at: str
    file_hash: str


class DocumentProcessor:
    """Processes various document types for RAG system."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Supported file types
        self.supported_extensions = {
            '.txt': self._load_text,
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.md': self._load_text,
        }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for change detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load text file."""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        except UnicodeDecodeError:
            # Try with different encoding
            loader = TextLoader(file_path, encoding='latin-1')
            return loader.load()
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file."""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load DOCX file."""
        loader = Docx2txtLoader(file_path)
        return loader.load()
    
    def process_file(self, file_path: str) -> tuple[List[Document], DocumentMetadata]:
        """Process a single file and return chunks with metadata."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        console.print(f"Processing {file_path.name}...", style="yellow")
        
        # Load document
        loader_func = self.supported_extensions[extension]
        documents = loader_func(str(file_path))
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add metadata to chunks
        file_hash = self._calculate_file_hash(str(file_path))
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'source_file': str(file_path),
                'file_type': extension,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'file_hash': file_hash
            })
            
            # Count tokens in chunk
            chunk_tokens = self._count_tokens(chunk.page_content)
            chunk.metadata['token_count'] = chunk_tokens
            total_tokens += chunk_tokens
        
        # Create metadata
        metadata = DocumentMetadata(
            file_path=str(file_path),
            file_type=extension,
            file_size=file_path.stat().st_size,
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            processed_at=str(file_path.stat().st_mtime),
            file_hash=file_hash
        )
        
        console.print(f"✅ Processed {file_path.name}: {len(chunks)} chunks, {total_tokens} tokens", style="green")
        
        return chunks, metadata
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> tuple[List[Document], List[DocumentMetadata]]:
        """Process all supported files in a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        all_files = []
        
        for extension in self.supported_extensions.keys():
            files = list(directory_path.glob(f"{pattern}{extension}"))
            all_files.extend(files)
        
        if not all_files:
            console.print(f"No supported files found in {directory_path}", style="yellow")
            return [], []
        
        console.print(f"Found {len(all_files)} files to process", style="cyan")
        
        all_chunks = []
        all_metadata = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing files...", total=len(all_files))
            
            for file_path in all_files:
                try:
                    chunks, metadata = self.process_file(file_path)
                    all_chunks.extend(chunks)
                    all_metadata.append(metadata)
                except Exception as e:
                    console.print(f"❌ Error processing {file_path.name}: {str(e)}", style="red")
                
                progress.advance(task)
        
        total_chunks = len(all_chunks)
        total_tokens = sum(meta.total_tokens for meta in all_metadata)
        
        console.print(f"✅ Processing complete: {total_chunks} chunks from {len(all_metadata)} files", style="bold green")
        console.print(f"📊 Total tokens: {total_tokens:,}", style="blue")
        
        return all_chunks, all_metadata
    
    def process_text(self, text: str, source: str = "direct_input") -> List[Document]:
        """Process raw text input."""
        # Create a document from the text
        document = Document(
            page_content=text,
            metadata={
                'source': source,
                'file_type': 'text',
                'token_count': self._count_tokens(text)
            }
        )
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([document])
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'token_count': self._count_tokens(chunk.page_content)
            })
        
        return chunks
    
    def get_processing_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed chunks."""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_tokens = sum(chunk.metadata.get('token_count', 0) for chunk in chunks)
        
        # Group by file type
        file_types = {}
        sources = set()
        
        for chunk in chunks:
            file_type = chunk.metadata.get('file_type', 'unknown')
            source = chunk.metadata.get('source_file', chunk.metadata.get('source', 'unknown'))
            
            file_types[file_type] = file_types.get(file_type, 0) + 1
            sources.add(source)
        
        return {
            'total_chunks': total_chunks,
            'total_tokens': total_tokens,
            'unique_sources': len(sources),
            'file_types': file_types,
            'avg_tokens_per_chunk': total_tokens / total_chunks if total_chunks > 0 else 0
        }


def create_sample_documents(output_dir: str = "sample_docs"):
    """Create sample documents for testing the RAG system."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Sample documents content
    documents = {
        "ai_overview.txt": """
Artificial Intelligence: A Comprehensive Overview

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The field of AI has evolved significantly since its inception in the 1950s.

Key Areas of AI:
1. Machine Learning - Algorithms that improve through experience
2. Natural Language Processing - Understanding and generating human language
3. Computer Vision - Interpreting visual information
4. Robotics - Physical AI systems that interact with the world
5. Expert Systems - AI that mimics human expertise in specific domains

Applications of AI:
- Healthcare: Diagnosis, drug discovery, personalized treatment
- Finance: Fraud detection, algorithmic trading, risk assessment
- Transportation: Autonomous vehicles, traffic optimization
- Entertainment: Recommendation systems, content generation
- Education: Personalized learning, intelligent tutoring systems

Challenges in AI:
- Ethical considerations and bias
- Data privacy and security
- Explainability and transparency
- Job displacement concerns
- Technical limitations and edge cases
        """,
        
        "machine_learning_guide.txt": """
Machine Learning: A Practical Guide

Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

Types of Machine Learning:

1. Supervised Learning
   - Uses labeled training data
   - Examples: Classification, Regression
   - Algorithms: Linear Regression, Decision Trees, Random Forest, SVM

2. Unsupervised Learning
   - Finds patterns in unlabeled data
   - Examples: Clustering, Dimensionality Reduction
   - Algorithms: K-Means, PCA, DBSCAN

3. Reinforcement Learning
   - Learns through interaction with environment
   - Uses rewards and penalties
   - Applications: Game playing, robotics, autonomous systems

Popular ML Frameworks:
- TensorFlow: Google's open-source platform
- PyTorch: Facebook's research-focused framework
- Scikit-learn: Python library for traditional ML
- Keras: High-level neural network API

Best Practices:
1. Data Quality: Clean, relevant, and sufficient data
2. Feature Engineering: Select and transform relevant features
3. Model Selection: Choose appropriate algorithms
4. Validation: Use cross-validation and test sets
5. Monitoring: Track model performance in production
        """,
        
        "rag_systems.txt": """
Retrieval-Augmented Generation (RAG) Systems

RAG systems combine the power of large language models with external knowledge retrieval to provide more accurate and up-to-date information.

How RAG Works:
1. Document Ingestion: Process and chunk documents
2. Embedding Creation: Convert text to vector representations
3. Vector Storage: Store embeddings in a vector database
4. Query Processing: Convert user queries to embeddings
5. Similarity Search: Find relevant document chunks
6. Context Augmentation: Add retrieved context to the prompt
7. Generation: LLM generates response with context

Components of RAG:
- Document Loader: Handles various file formats
- Text Splitter: Breaks documents into manageable chunks
- Embedding Model: Creates vector representations
- Vector Store: Stores and retrieves embeddings
- Retriever: Finds relevant documents
- LLM: Generates final responses

Benefits of RAG:
- Access to current information
- Reduced hallucinations
- Transparent source attribution
- Domain-specific knowledge integration
- Cost-effective compared to fine-tuning

Challenges:
- Chunk size optimization
- Retrieval quality
- Context window limitations
- Embedding model selection
- Performance optimization
        """
    }
    
    # Create the documents
    for filename, content in documents.items():
        file_path = output_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    console.print(f"✅ Created {len(documents)} sample documents in {output_dir}/", style="green")
    return output_path
