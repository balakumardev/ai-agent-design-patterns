"""Tool definitions for the tool-using agent."""

import json
import requests
from typing import Dict, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate")


class Calculator(BaseTool):
    """A simple calculator tool for mathematical operations."""
    
    name: str = "calculator"
    description: str = "Performs mathematical calculations. Input should be a valid mathematical expression."
    args_schema: type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Execute the calculator tool."""
        try:
            # Basic security: only allow safe mathematical operations
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression. Only numbers, +, -, *, /, (, ), and spaces are allowed."
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating expression '{expression}': {str(e)}"


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="Search query to look up on the web")


class WebSearch(BaseTool):
    """A web search tool using a simple HTTP search service."""
    
    name: str = "web_search"
    description: str = "Searches the web for information. Input should be a search query string."
    args_schema: type[BaseModel] = WebSearchInput
    
    def _run(self, query: str) -> str:
        """Execute the web search tool."""
        try:
            # Using DuckDuckGo's instant answer API as a simple search
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('Answer'):
                return f"Answer: {data['Answer']}"
            elif data.get('AbstractText'):
                return f"Information: {data['AbstractText']}"
            elif data.get('Definition'):
                return f"Definition: {data['Definition']}"
            else:
                return f"Search completed for '{query}' but no specific answer found. You may want to try a more specific query."
                
        except Exception as e:
            return f"Error searching for '{query}': {str(e)}"


class FileOperationsInput(BaseModel):
    """Input schema for file operations tool."""
    operation: str = Field(description="Operation to perform: 'write', 'read', or 'list'")
    filename: str = Field(description="Name of the file to operate on", default="")
    content: str = Field(description="Content to write to file (for write operation)", default="")


class FileOperations(BaseTool):
    """A tool for basic file operations."""
    
    name: str = "file_operations"
    description: str = "Performs file operations like read, write, or list files. Operations: 'write', 'read', 'list'"
    args_schema: type[BaseModel] = FileOperationsInput
    
    def _run(self, operation: str, filename: str = "", content: str = "") -> str:
        """Execute the file operations tool."""
        try:
            if operation == "write":
                if not filename:
                    return "Error: Filename required for write operation"
                with open(filename, 'w') as f:
                    f.write(content)
                return f"Successfully wrote content to {filename}"
                
            elif operation == "read":
                if not filename:
                    return "Error: Filename required for read operation"
                with open(filename, 'r') as f:
                    file_content = f.read()
                return f"Content of {filename}:\n{file_content}"
                
            elif operation == "list":
                import os
                files = os.listdir('.')
                return f"Files in current directory: {', '.join(files)}"
                
            else:
                return f"Error: Unknown operation '{operation}'. Available operations: write, read, list"
                
        except Exception as e:
            return f"Error performing file operation '{operation}': {str(e)}"


def get_tools():
    """Return a list of available tools."""
    return [
        Calculator(),
        WebSearch(),
        FileOperations()
    ]