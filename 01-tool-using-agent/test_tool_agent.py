"""Tests for the tool-using agent."""

import pytest
import os
from unittest.mock import patch, MagicMock
from tool_agent import ToolUsingAgent
from tools import Calculator, WebSearch, FileOperations


class TestCalculator:
    """Test the Calculator tool."""
    
    def test_calculator_basic_operations(self):
        """Test basic mathematical operations."""
        calc = Calculator()
        
        # Test addition
        result = calc._run("2 + 3")
        assert "Result: 5" in result
        
        # Test multiplication
        result = calc._run("4 * 5")
        assert "Result: 20" in result
        
        # Test complex expression
        result = calc._run("(10 + 5) * 2")
        assert "Result: 30" in result
    
    def test_calculator_invalid_input(self):
        """Test calculator with invalid input."""
        calc = Calculator()
        
        # Test invalid characters
        result = calc._run("2 + import os")
        assert "Error: Invalid characters" in result
        
        # Test malformed expression
        result = calc._run("2 + * 3")
        assert "Error calculating" in result
    
    def test_calculator_security(self):
        """Test calculator security measures."""
        calc = Calculator()
        
        # Test that dangerous operations are blocked
        dangerous_inputs = [
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('1+1')",
            "open('/etc/passwd')"
        ]
        
        for dangerous_input in dangerous_inputs:
            result = calc._run(dangerous_input)
            assert "Error: Invalid characters" in result


class TestWebSearch:
    """Test the WebSearch tool."""
    
    @patch('tools.requests.get')
    def test_web_search_with_answer(self, mock_get):
        """Test web search when API returns an answer."""
        # Mock successful response with answer
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'Answer': 'The capital of France is Paris'
        }
        mock_get.return_value = mock_response
        
        search = WebSearch()
        result = search._run("capital of France")
        
        assert "Answer: The capital of France is Paris" in result
        mock_get.assert_called_once()
    
    @patch('tools.requests.get')
    def test_web_search_with_abstract(self, mock_get):
        """Test web search when API returns abstract text."""
        # Mock response with abstract text
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'AbstractText': 'France is a country in Europe...'
        }
        mock_get.return_value = mock_response
        
        search = WebSearch()
        result = search._run("France")
        
        assert "Information: France is a country in Europe..." in result
    
    @patch('tools.requests.get')
    def test_web_search_no_results(self, mock_get):
        """Test web search when no results are found."""
        # Mock response with no useful data
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        search = WebSearch()
        result = search._run("very obscure query")
        
        assert "no specific answer found" in result
    
    @patch('tools.requests.get')
    def test_web_search_error(self, mock_get):
        """Test web search when an error occurs."""
        # Mock request exception
        mock_get.side_effect = Exception("Network error")
        
        search = WebSearch()
        result = search._run("test query")
        
        assert "Error searching" in result


class TestFileOperations:
    """Test the FileOperations tool."""
    
    def test_file_write_and_read(self):
        """Test writing and reading files."""
        file_ops = FileOperations()
        test_filename = "test_file.txt"
        test_content = "Hello, World!"
        
        try:
            # Test write operation
            result = file_ops._run("write", test_filename, test_content)
            assert f"Successfully wrote content to {test_filename}" in result
            
            # Test read operation
            result = file_ops._run("read", test_filename)
            assert f"Content of {test_filename}:" in result
            assert test_content in result
            
        finally:
            # Clean up
            if os.path.exists(test_filename):
                os.remove(test_filename)
    
    def test_file_list_operation(self):
        """Test listing files in directory."""
        file_ops = FileOperations()
        
        result = file_ops._run("list")
        assert "Files in current directory:" in result
    
    def test_file_operations_errors(self):
        """Test file operations error handling."""
        file_ops = FileOperations()
        
        # Test write without filename
        result = file_ops._run("write", "", "content")
        assert "Error: Filename required for write operation" in result
        
        # Test read without filename
        result = file_ops._run("read", "")
        assert "Error: Filename required for read operation" in result
        
        # Test invalid operation
        result = file_ops._run("invalid_op")
        assert "Error: Unknown operation" in result
        
        # Test reading non-existent file
        result = file_ops._run("read", "non_existent_file.txt")
        assert "Error performing file operation" in result


class TestToolUsingAgent:
    """Test the ToolUsingAgent class."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        agent = ToolUsingAgent()
        
        assert agent.tools is not None
        assert len(agent.tools) == 3
        assert agent.model is not None
        assert agent.graph is not None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_agent_tools_available(self):
        """Test that all expected tools are available."""
        agent = ToolUsingAgent()
        
        tool_names = [tool.name for tool in agent.tools]
        expected_tools = ['calculator', 'web_search', 'file_operations']
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('tool_agent.ChatOpenAI')
    def test_agent_error_handling(self, mock_chat_openai):
        """Test agent error handling."""
        # Mock the model to raise an exception
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.invoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_model
        
        agent = ToolUsingAgent()
        result = agent.run("test query")
        
        assert "Error processing query" in result


if __name__ == "__main__":
    pytest.main([__file__])
