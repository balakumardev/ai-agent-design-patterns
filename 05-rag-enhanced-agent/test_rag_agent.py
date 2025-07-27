"""Tests for the RAG-enhanced agent."""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from document_processor import DocumentProcessor, DocumentMetadata, create_sample_documents
from vector_store import VectorStoreManager, VectorStoreConfig, SearchResult
from rag_agent import RAGEnhancedAgent, RAGResponse


class TestDocumentProcessor:
    """Test the DocumentProcessor class."""
    
    def test_processor_initialization(self):
        """Test document processor initialization."""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100
        assert processor.text_splitter is not None
        assert processor.encoding is not None
    
    def test_process_text(self):
        """Test processing raw text."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        text = "This is a test document. " * 20  # Create text longer than chunk size
        chunks = processor.process_text(text, "test_source")
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(chunk.metadata['source'] == 'test_source' for chunk in chunks)
        assert all('token_count' in chunk.metadata for chunk in chunks)
    
    def test_process_file_text(self):
        """Test processing a text file."""
        processor = DocumentProcessor()
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for processing.\n" * 10)
            temp_file = f.name
        
        try:
            chunks, metadata = processor.process_file(temp_file)
            
            assert len(chunks) > 0
            assert isinstance(metadata, DocumentMetadata)
            assert metadata.file_type == '.txt'
            assert metadata.chunk_count == len(chunks)
            assert all('source_file' in chunk.metadata for chunk in chunks)
            
        finally:
            os.unlink(temp_file)
    
    def test_process_nonexistent_file(self):
        """Test processing a non-existent file."""
        processor = DocumentProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.process_file("nonexistent_file.txt")
    
    def test_process_unsupported_file(self):
        """Test processing an unsupported file type."""
        processor = DocumentProcessor()
        
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                processor.process_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        processor = DocumentProcessor()
        
        # Create some test chunks
        chunks = processor.process_text("Test document content", "test_source")
        
        stats = processor.get_processing_stats(chunks)
        
        assert 'total_chunks' in stats
        assert 'total_tokens' in stats
        assert 'unique_sources' in stats
        assert 'file_types' in stats
        assert 'avg_tokens_per_chunk' in stats
        assert stats['total_chunks'] == len(chunks)
    
    def test_create_sample_documents(self):
        """Test creating sample documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = create_sample_documents(temp_dir)
            
            assert sample_path.exists()
            assert sample_path.is_dir()
            
            # Check that files were created
            files = list(sample_path.glob("*.txt"))
            assert len(files) > 0
            
            # Check file content
            for file_path in files:
                content = file_path.read_text()
                assert len(content) > 0


class TestVectorStoreConfig:
    """Test the VectorStoreConfig class."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = VectorStoreConfig()
        
        assert config.collection_name == "rag_documents"
        assert config.persist_directory == "./chroma_db"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.similarity_threshold == 0.7
    
    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = VectorStoreConfig(
            collection_name="test_collection",
            chunk_size=500,
            similarity_threshold=0.8
        )
        
        assert config.collection_name == "test_collection"
        assert config.chunk_size == 500
        assert config.similarity_threshold == 0.8


class TestSearchResult:
    """Test the SearchResult class."""
    
    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            content="Test content",
            metadata={"source": "test.txt"},
            similarity_score=0.85,
            source="test.txt",
            chunk_index=0
        )
        
        assert result.content == "Test content"
        assert result.metadata == {"source": "test.txt"}
        assert result.similarity_score == 0.85
        assert result.source == "test.txt"
        assert result.chunk_index == 0


class TestVectorStoreManager:
    """Test the VectorStoreManager class."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_manager_initialization(self):
        """Test vector store manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(persist_directory=temp_dir)
            
            with patch('vector_store.OpenAIEmbeddings'), \
                 patch('vector_store.chromadb.PersistentClient'), \
                 patch('vector_store.Chroma'):
                
                manager = VectorStoreManager(config)
                assert manager.config == config
                assert manager.persist_directory == Path(temp_dir)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('vector_store.OpenAIEmbeddings')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.Chroma')
    def test_add_documents(self, mock_chroma, mock_client, mock_embeddings):
        """Test adding documents to vector store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(persist_directory=temp_dir)
            manager = VectorStoreManager(config)
            
            # Mock vector store
            mock_vector_store = MagicMock()
            mock_vector_store.add_documents.return_value = ['id1', 'id2']
            manager.vector_store = mock_vector_store
            
            # Create test documents
            processor = DocumentProcessor()
            chunks = processor.process_text("Test document content", "test")
            
            # Add documents
            document_ids = manager.add_documents(chunks, show_progress=False)
            
            assert len(document_ids) == 2
            mock_vector_store.add_documents.assert_called_once_with(chunks)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('vector_store.OpenAIEmbeddings')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.Chroma')
    def test_similarity_search(self, mock_chroma, mock_client, mock_embeddings):
        """Test similarity search."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(persist_directory=temp_dir)
            manager = VectorStoreManager(config)
            
            # Mock vector store
            mock_vector_store = MagicMock()
            mock_doc = MagicMock()
            mock_doc.page_content = "Test content"
            mock_doc.metadata = {"source": "test.txt", "chunk_index": 0}
            mock_vector_store.similarity_search_with_score.return_value = [(mock_doc, 0.2)]
            manager.vector_store = mock_vector_store
            
            # Perform search
            results = manager.similarity_search("test query", k=5)
            
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].content == "Test content"
            assert results[0].similarity_score == 0.8  # 1 - 0.2


class TestRAGEnhancedAgent:
    """Test the RAGEnhancedAgent class."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('rag_agent.VectorStoreManager')
    @patch('rag_agent.DocumentProcessor')
    def test_agent_initialization(self, mock_processor, mock_vector_store):
        """Test RAG agent initialization."""
        agent = RAGEnhancedAgent()
        
        assert agent.model is not None
        assert agent.vector_store_config is not None
        assert agent.graph is not None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('rag_agent.ChatOpenAI')
    @patch('rag_agent.VectorStoreManager')
    @patch('rag_agent.DocumentProcessor')
    def test_query_processing(self, mock_processor, mock_vector_store, mock_chat_openai):
        """Test query processing."""
        # Mock the model response
        mock_response = MagicMock()
        mock_response.content = "This is a test response based on the context."
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model
        
        # Mock vector store
        mock_vs_instance = MagicMock()
        mock_vs_instance.similarity_search.return_value = []
        mock_vs_instance.get_relevant_context.return_value = "Test context"
        mock_vector_store.return_value = mock_vs_instance
        
        agent = RAGEnhancedAgent()
        response = agent.query("What is artificial intelligence?")
        
        assert isinstance(response, RAGResponse)
        assert response.query == "What is artificial intelligence?"
        assert len(response.answer) > 0
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('rag_agent.VectorStoreManager')
    @patch('rag_agent.DocumentProcessor')
    def test_add_text(self, mock_processor, mock_vector_store):
        """Test adding text to knowledge base."""
        # Mock document processor
        mock_proc_instance = MagicMock()
        mock_proc_instance.process_text.return_value = [MagicMock()]
        mock_processor.return_value = mock_proc_instance
        
        # Mock vector store
        mock_vs_instance = MagicMock()
        mock_vector_store.return_value = mock_vs_instance
        
        agent = RAGEnhancedAgent()
        agent.add_text("Test text content", "test_source")
        
        mock_proc_instance.process_text.assert_called_once_with("Test text content", "test_source")
        mock_vs_instance.add_documents.assert_called_once()


class TestRAGResponse:
    """Test the RAGResponse class."""
    
    def test_rag_response_creation(self):
        """Test creating a RAG response."""
        sources = [SearchResult(
            content="Test content",
            metadata={},
            similarity_score=0.8,
            source="test.txt",
            chunk_index=0
        )]
        
        response = RAGResponse(
            answer="Test answer",
            sources=sources,
            context_used="Test context",
            confidence_score=0.85,
            query="Test query"
        )
        
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.context_used == "Test context"
        assert response.confidence_score == 0.85
        assert response.query == "Test query"


if __name__ == "__main__":
    pytest.main([__file__])
