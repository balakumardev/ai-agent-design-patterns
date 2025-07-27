"""Tests for the reflection agent."""

import pytest
import os
from unittest.mock import patch, MagicMock
from reflection_agent import ReflectionAgent, ReflectionResult, GraphState


class TestReflectionAgent:
    """Test the ReflectionAgent class."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        agent = ReflectionAgent(max_iterations=3)
        
        assert agent.model is not None
        assert agent.max_iterations == 3
        assert agent.graph is not None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('reflection_agent.ChatOpenAI')
    def test_generate_response(self, mock_chat_openai):
        """Test the generate_response method."""
        # Mock the model response
        mock_response = MagicMock()
        mock_response.content = "This is a generated response."
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model
        
        agent = ReflectionAgent()
        
        state = {
            "original_query": "Test query",
            "current_response": "",
            "critique": "",
            "iteration": 0,
            "max_iterations": 3,
            "quality_threshold": 0.8,
            "final_response": ""
        }
        
        result = agent.generate_response(state)
        
        assert result["current_response"] == "This is a generated response."
        assert result["iteration"] == 1
        mock_model.invoke.assert_called_once()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('reflection_agent.ChatOpenAI')
    def test_reflect_on_response(self, mock_chat_openai):
        """Test the reflect_on_response method."""
        # Mock the model response
        mock_response = MagicMock()
        mock_response.content = "This response could be improved by adding more examples."
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model
        
        agent = ReflectionAgent()
        
        state = {
            "original_query": "Explain machine learning",
            "current_response": "Machine learning is a subset of AI.",
            "critique": "",
            "iteration": 1,
            "max_iterations": 3,
            "quality_threshold": 0.8,
            "final_response": ""
        }
        
        result = agent.reflect_on_response(state)
        
        assert result["critique"] == "This response could be improved by adding more examples."
        mock_model.invoke.assert_called_once()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('reflection_agent.ChatOpenAI')
    def test_revise_response(self, mock_chat_openai):
        """Test the revise_response method."""
        # Mock the model response
        mock_response = MagicMock()
        mock_response.content = "Machine learning is a subset of AI that enables computers to learn from data."
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model
        
        agent = ReflectionAgent()
        
        state = {
            "original_query": "Explain machine learning",
            "current_response": "Machine learning is a subset of AI.",
            "critique": "This response could be improved by adding more examples.",
            "iteration": 1,
            "max_iterations": 3,
            "quality_threshold": 0.8,
            "final_response": ""
        }
        
        result = agent.revise_response(state)
        
        assert result["current_response"] == "Machine learning is a subset of AI that enables computers to learn from data."
        assert result["iteration"] == 2
        mock_model.invoke.assert_called_once()
    
    def test_should_revise_with_improvements_needed(self):
        """Test should_revise when improvements are needed."""
        agent = ReflectionAgent()
        
        state = {
            "original_query": "Test query",
            "current_response": "Test response",
            "critique": "This response could be improved by adding more details.",
            "iteration": 1,
            "max_iterations": 3,
            "quality_threshold": 0.8,
            "final_response": ""
        }
        
        result = agent.should_revise(state)
        assert result == "revise"
    
    def test_should_revise_with_no_improvements_needed(self):
        """Test should_revise when no improvements are needed."""
        agent = ReflectionAgent()
        
        state = {
            "original_query": "Test query",
            "current_response": "Test response",
            "critique": "This response is excellent and comprehensive.",
            "iteration": 1,
            "max_iterations": 3,
            "quality_threshold": 0.8,
            "final_response": ""
        }
        
        result = agent.should_revise(state)
        assert result == "end"
    
    def test_should_revise_max_iterations_reached(self):
        """Test should_revise when max iterations are reached."""
        agent = ReflectionAgent()
        
        state = {
            "original_query": "Test query",
            "current_response": "Test response",
            "critique": "This response could be improved.",
            "iteration": 3,
            "max_iterations": 3,
            "quality_threshold": 0.8,
            "final_response": ""
        }
        
        result = agent.should_revise(state)
        assert result == "end"
    
    def test_should_revise_improvement_keywords(self):
        """Test should_revise with various improvement keywords."""
        agent = ReflectionAgent()
        
        improvement_phrases = [
            "should include more examples",
            "missing important details",
            "unclear explanation",
            "needs better structure",
            "recommend adding",
            "suggest improving",
            "better organization needed",
            "enhance the content",
            "expand on this topic"
        ]
        
        for phrase in improvement_phrases:
            state = {
                "original_query": "Test query",
                "current_response": "Test response",
                "critique": f"The response is good but {phrase}.",
                "iteration": 1,
                "max_iterations": 3,
                "quality_threshold": 0.8,
                "final_response": ""
            }
            
            result = agent.should_revise(state)
            assert result == "revise", f"Failed for phrase: {phrase}"
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('reflection_agent.ChatOpenAI')
    def test_run_method_success(self, mock_chat_openai):
        """Test the run method with successful execution."""
        # Mock the model responses
        mock_responses = [
            MagicMock(content="Initial response"),
            MagicMock(content="This response is excellent and needs no improvement."),
        ]
        mock_model = MagicMock()
        mock_model.invoke.side_effect = mock_responses
        mock_chat_openai.return_value = mock_model
        
        agent = ReflectionAgent()
        result = agent.run("Test query", max_iterations=2)
        
        assert isinstance(result, ReflectionResult)
        assert result.content == "Initial response"
        assert result.iteration >= 1
        assert result.critique is not None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('reflection_agent.ChatOpenAI')
    def test_run_method_error_handling(self, mock_chat_openai):
        """Test the run method with error handling."""
        # Mock the model to raise an exception
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_model
        
        agent = ReflectionAgent()
        result = agent.run("Test query")
        
        assert isinstance(result, ReflectionResult)
        assert "Error processing query" in result.content
        assert result.iteration == 0
        assert not result.needs_improvement


class TestReflectionResult:
    """Test the ReflectionResult dataclass."""
    
    def test_reflection_result_creation(self):
        """Test creating a ReflectionResult."""
        result = ReflectionResult(
            content="Test content",
            critique="Test critique",
            needs_improvement=True,
            iteration=2
        )
        
        assert result.content == "Test content"
        assert result.critique == "Test critique"
        assert result.needs_improvement is True
        assert result.iteration == 2
    
    def test_reflection_result_defaults(self):
        """Test ReflectionResult with minimal parameters."""
        result = ReflectionResult(
            content="Test content",
            critique="Test critique",
            needs_improvement=False,
            iteration=1
        )
        
        assert result.content == "Test content"
        assert result.critique == "Test critique"
        assert result.needs_improvement is False
        assert result.iteration == 1


if __name__ == "__main__":
    pytest.main([__file__])
