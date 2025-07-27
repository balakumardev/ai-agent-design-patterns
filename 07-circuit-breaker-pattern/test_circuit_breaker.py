"""Tests for the circuit breaker implementation."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CallResult, CircuitState, FailureType,
    SimpleFallback, CachedFallback, DegradedServiceFallback
)
from resilient_agent import ResilientAgent, AgentResponse


class TestCircuitBreakerConfig:
    """Test the CircuitBreakerConfig class."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout_duration == 30.0
        assert config.quality_threshold == 0.7
        assert config.rate_limit_per_minute == 60
        assert config.sliding_window_size == 100
        assert config.half_open_max_calls == 5
    
    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            quality_threshold=0.8
        )
        
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.quality_threshold == 0.8


class TestCallResult:
    """Test the CallResult class."""
    
    def test_call_result_creation(self):
        """Test creating a call result."""
        result = CallResult(
            success=True,
            response="test response",
            duration=1.5,
            quality_score=0.8
        )
        
        assert result.success is True
        assert result.response == "test response"
        assert result.duration == 1.5
        assert result.quality_score == 0.8
        assert result.error is None
        assert result.failure_type is None
    
    def test_call_result_failure(self):
        """Test creating a failure call result."""
        error = Exception("Test error")
        result = CallResult(
            success=False,
            error=error,
            failure_type=FailureType.ERROR,
            duration=0.5
        )
        
        assert result.success is False
        assert result.error == error
        assert result.failure_type == FailureType.ERROR
        assert result.duration == 0.5


class TestFallbackStrategies:
    """Test fallback strategy implementations."""
    
    def test_simple_fallback(self):
        """Test simple fallback strategy."""
        fallback = SimpleFallback("fallback response")
        
        result = fallback.execute()
        assert result == "fallback response"
    
    def test_cached_fallback(self):
        """Test cached fallback strategy."""
        fallback = CachedFallback()
        fallback.add_to_cache("key1", "cached response 1")
        fallback.add_to_cache("key2", "cached response 2")
        
        result1 = fallback.execute(cache_key="key1")
        result2 = fallback.execute(cache_key="key2")
        result3 = fallback.execute(cache_key="nonexistent")
        
        assert result1 == "cached response 1"
        assert result2 == "cached response 2"
        assert result3 == "Service temporarily unavailable"
    
    def test_degraded_service_fallback(self):
        """Test degraded service fallback strategy."""
        def degraded_function(x, y):
            return x + y
        
        fallback = DegradedServiceFallback(degraded_function)
        
        result = fallback.execute(5, 3)
        assert result == 8


class TestCircuitBreaker:
    """Test the CircuitBreaker class."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig(failure_threshold=3)
        fallback = SimpleFallback("fallback")
        
        cb = CircuitBreaker("test", config, fallback)
        
        assert cb.name == "test"
        assert cb.config == config
        assert cb.fallback == fallback
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0
        assert cb.consecutive_successes == 0
    
    def test_successful_call(self):
        """Test successful function call."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        def successful_function():
            return "success"
        
        result = cb.call(successful_function)
        
        assert result.success is True
        assert result.response == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0
        assert cb.consecutive_successes == 1
    
    def test_failed_call(self):
        """Test failed function call."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)
        
        def failing_function():
            raise Exception("Test error")
        
        # First failure
        result1 = cb.call(failing_function)
        assert result1.success is False
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 1
        
        # Second failure should open circuit
        result2 = cb.call(failing_function)
        assert result2.success is False
        assert cb.state == CircuitState.OPEN
        assert cb.consecutive_failures == 2
    
    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        def failing_function():
            raise Exception("Test error")
        
        # Fail up to threshold
        for i in range(3):
            result = cb.call(failing_function)
            assert result.success is False
        
        assert cb.state == CircuitState.OPEN
    
    def test_circuit_blocks_when_open(self):
        """Test circuit blocks calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        fallback = SimpleFallback("fallback response")
        cb = CircuitBreaker("test", config, fallback)
        
        def failing_function():
            raise Exception("Test error")
        
        # Open the circuit
        cb.call(failing_function)
        assert cb.state == CircuitState.OPEN
        
        # Next call should use fallback
        def normal_function():
            return "normal response"
        
        result = cb.call(normal_function)
        assert result.success is True
        assert result.response == "fallback response"
    
    def test_circuit_half_open_transition(self):
        """Test transition to half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)
        
        def failing_function():
            raise Exception("Test error")
        
        # Open the circuit
        cb.call(failing_function)
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        def successful_function():
            return "success"
        
        # Should transition to half-open
        result = cb.call(successful_function)
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_circuit_closes_after_successes(self):
        """Test circuit closes after enough successes in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1, 
            recovery_timeout=0.1,
            success_threshold=2
        )
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        def failing_function():
            raise Exception("Test error")
        
        cb.call(failing_function)
        assert cb.state == CircuitState.OPEN
        
        # Wait and transition to half-open
        time.sleep(0.2)
        
        def successful_function():
            return "success"
        
        # First success in half-open
        result1 = cb.call(successful_function)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second success should close circuit
        result2 = cb.call(successful_function)
        assert cb.state == CircuitState.CLOSED
    
    def test_quality_evaluation(self):
        """Test quality evaluation functionality."""
        config = CircuitBreakerConfig(quality_threshold=0.7)
        cb = CircuitBreaker("test", config)
        
        def response_function():
            return "low quality response"
        
        def quality_evaluator(response):
            return 0.5  # Below threshold
        
        result = cb.call(response_function, quality_evaluator=quality_evaluator)
        
        assert result.success is False
        assert result.failure_type == FailureType.QUALITY_THRESHOLD
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        config = CircuitBreakerConfig(rate_limit_per_minute=2)
        cb = CircuitBreaker("test", config)
        
        def test_function():
            return "response"
        
        # First two calls should succeed
        result1 = cb.call(test_function)
        result2 = cb.call(test_function)
        
        assert result1.success is True
        assert result2.success is True
        
        # Third call should be rate limited
        result3 = cb.call(test_function)
        assert result3.success is False
        assert result3.failure_type == FailureType.RATE_LIMIT
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker("test", config)
        
        def successful_function():
            time.sleep(0.01)  # Small delay for timing
            return "success"
        
        def failing_function():
            raise Exception("Test error")
        
        # Make some calls
        cb.call(successful_function)
        cb.call(successful_function)
        cb.call(failing_function)
        
        metrics = cb.get_metrics()
        
        assert metrics.total_calls == 3
        assert metrics.successful_calls == 2
        assert metrics.failed_calls == 1
        assert metrics.success_rate == 2/3
        assert metrics.average_response_time > 0
    
    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        
        def failing_function():
            raise Exception("Test error")
        
        # Open the circuit
        cb.call(failing_function)
        assert cb.state == CircuitState.OPEN
        
        # Reset manually
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0
    
    def test_force_open(self):
        """Test manually forcing circuit open."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker("test", config)
        
        assert cb.state == CircuitState.CLOSED
        
        cb.force_open()
        assert cb.state == CircuitState.OPEN


class TestResilientAgent:
    """Test the ResilientAgent class."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_agent_initialization(self):
        """Test resilient agent initialization."""
        agent = ResilientAgent()
        
        assert agent.model is not None
        assert len(agent.circuit_breakers) > 0
        assert "llm" in agent.circuit_breakers
        assert agent.graph is not None
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('resilient_agent.ChatOpenAI')
    def test_query_processing(self, mock_chat_openai):
        """Test query processing with circuit breaker."""
        # Mock the model response
        mock_response = MagicMock()
        mock_response.content = "This is a test response."
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model
        
        agent = ResilientAgent()
        response = agent.query("What is AI?")
        
        assert isinstance(response, AgentResponse)
        assert response.success is True
        assert len(response.content) > 0
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_circuit_breaker_status(self):
        """Test getting circuit breaker status."""
        agent = ResilientAgent()
        
        status = agent.get_circuit_breaker_status()
        
        assert isinstance(status, dict)
        assert "llm" in status
        assert "state" in status["llm"]
        assert "success_rate" in status["llm"]
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_simulate_service_degradation(self):
        """Test simulating service degradation."""
        agent = ResilientAgent()
        
        # Initially closed
        assert agent.circuit_breakers["llm"].state == CircuitState.CLOSED
        
        # Simulate failure
        agent.simulate_service_degradation("llm")
        assert agent.circuit_breakers["llm"].state == CircuitState.OPEN
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_reset_circuit_breaker(self):
        """Test resetting circuit breaker."""
        agent = ResilientAgent()
        
        # Force open
        agent.circuit_breakers["llm"].force_open()
        assert agent.circuit_breakers["llm"].state == CircuitState.OPEN
        
        # Reset
        agent.reset_circuit_breaker("llm")
        assert agent.circuit_breakers["llm"].state == CircuitState.CLOSED


class TestAgentResponse:
    """Test the AgentResponse class."""
    
    def test_agent_response_creation(self):
        """Test creating an agent response."""
        response = AgentResponse(
            content="Test response",
            success=True,
            used_fallback=False,
            circuit_state=CircuitState.CLOSED,
            response_time=1.5,
            quality_score=0.8
        )
        
        assert response.content == "Test response"
        assert response.success is True
        assert response.used_fallback is False
        assert response.circuit_state == CircuitState.CLOSED
        assert response.response_time == 1.5
        assert response.quality_score == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
