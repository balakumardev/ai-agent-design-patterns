"""Circuit breaker implementation for resilient agent systems."""

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics
from abc import ABC, abstractmethod


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing if service recovered


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker."""
    TIMEOUT = "timeout"
    ERROR = "error"
    RATE_LIMIT = "rate_limit"
    QUALITY_THRESHOLD = "quality_threshold"
    CUSTOM = "custom"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout_duration: float = 30.0     # Request timeout in seconds
    quality_threshold: float = 0.7      # Minimum quality score (0-1)
    rate_limit_per_minute: int = 60     # Maximum requests per minute
    sliding_window_size: int = 100      # Size of sliding window for metrics
    half_open_max_calls: int = 5        # Max calls allowed in half-open state


@dataclass
class CallResult:
    """Result of a circuit breaker protected call."""
    success: bool
    response: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    quality_score: Optional[float] = None
    failure_type: Optional[FailureType] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeouts: int = 0
    rate_limited: int = 0
    quality_failures: int = 0
    average_response_time: float = 0.0
    success_rate: float = 0.0
    current_state: CircuitState = CircuitState.CLOSED
    state_changed_at: datetime = field(default_factory=datetime.now)
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the fallback strategy."""
        pass


class SimpleFallback(FallbackStrategy):
    """Simple fallback that returns a predefined response."""
    
    def __init__(self, response: Any):
        self.response = response
    
    def execute(self, *args, **kwargs) -> Any:
        return self.response


class CachedFallback(FallbackStrategy):
    """Fallback that returns cached responses."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
    
    def add_to_cache(self, key: str, value: Any):
        """Add a value to the cache."""
        self.cache[key] = value
    
    def execute(self, cache_key: str = "default", *args, **kwargs) -> Any:
        return self.cache.get(cache_key, "Service temporarily unavailable")


class DegradedServiceFallback(FallbackStrategy):
    """Fallback that provides degraded service functionality."""
    
    def __init__(self, degraded_function: Callable):
        self.degraded_function = degraded_function
    
    def execute(self, *args, **kwargs) -> Any:
        return self.degraded_function(*args, **kwargs)


class CircuitBreaker:
    """Circuit breaker implementation for protecting against cascading failures."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig, fallback: Optional[FallbackStrategy] = None):
        self.name = name
        self.config = config
        self.fallback = fallback or SimpleFallback("Service temporarily unavailable")
        
        # State management
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.last_failure_time: Optional[datetime] = None
        
        # Counters
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.half_open_calls = 0
        
        # Metrics
        self.call_history: List[CallResult] = []
        self.rate_limit_window: List[datetime] = []
        
        # Thread safety
        self.lock = threading.RLock()
    
    def call(self, func: Callable, *args, quality_evaluator: Optional[Callable[[Any], float]] = None, **kwargs) -> CallResult:
        """Execute a function call with circuit breaker protection."""
        with self.lock:
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    return self._execute_fallback(*args, **kwargs)
            
            # Check rate limiting
            if not self._check_rate_limit():
                return CallResult(
                    success=False,
                    error=Exception("Rate limit exceeded"),
                    failure_type=FailureType.RATE_LIMIT
                )
            
            # Check half-open state limits
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    return self._execute_fallback(*args, **kwargs)
                self.half_open_calls += 1
            
            # Execute the function call
            return self._execute_call(func, *args, quality_evaluator=quality_evaluator, **kwargs)
    
    def _execute_call(self, func: Callable, *args, quality_evaluator: Optional[Callable[[Any], float]] = None, **kwargs) -> CallResult:
        """Execute the actual function call with monitoring."""
        start_time = time.time()
        
        try:
            # Set timeout if configured
            if hasattr(func, '__timeout__'):
                # Function has custom timeout handling
                response = func(*args, **kwargs)
            else:
                # Simple timeout implementation
                response = func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            # Check for timeout
            if duration > self.config.timeout_duration:
                return self._handle_failure(
                    Exception(f"Call timed out after {duration:.2f}s"),
                    duration,
                    FailureType.TIMEOUT
                )
            
            # Evaluate quality if evaluator provided
            quality_score = None
            if quality_evaluator:
                try:
                    quality_score = quality_evaluator(response)
                    if quality_score < self.config.quality_threshold:
                        return self._handle_failure(
                            Exception(f"Quality score {quality_score:.2f} below threshold {self.config.quality_threshold}"),
                            duration,
                            FailureType.QUALITY_THRESHOLD
                        )
                except Exception as e:
                    # Quality evaluation failed, but don't fail the call
                    quality_score = None
            
            # Success
            return self._handle_success(response, duration, quality_score)
            
        except Exception as e:
            duration = time.time() - start_time
            return self._handle_failure(e, duration, FailureType.ERROR)
    
    def _handle_success(self, response: Any, duration: float, quality_score: Optional[float] = None) -> CallResult:
        """Handle successful call."""
        result = CallResult(
            success=True,
            response=response,
            duration=duration,
            quality_score=quality_score
        )
        
        self.call_history.append(result)
        self._trim_call_history()
        
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        
        # Transition from half-open to closed if enough successes
        if (self.state == CircuitState.HALF_OPEN and 
            self.consecutive_successes >= self.config.success_threshold):
            self._transition_to_closed()
        
        return result
    
    def _handle_failure(self, error: Exception, duration: float, failure_type: FailureType) -> CallResult:
        """Handle failed call."""
        result = CallResult(
            success=False,
            error=error,
            duration=duration,
            failure_type=failure_type
        )
        
        self.call_history.append(result)
        self._trim_call_history()
        
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()
        
        # Check if we should open the circuit
        if (self.state == CircuitState.CLOSED and 
            self.consecutive_failures >= self.config.failure_threshold):
            self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state goes back to open
            self._transition_to_open()
        
        return result
    
    def _execute_fallback(self, *args, **kwargs) -> CallResult:
        """Execute fallback strategy."""
        try:
            response = self.fallback.execute(*args, **kwargs)
            return CallResult(
                success=True,
                response=response,
                duration=0.0
            )
        except Exception as e:
            return CallResult(
                success=False,
                error=e,
                duration=0.0
            )
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Remove old entries
        self.rate_limit_window = [ts for ts in self.rate_limit_window if ts > cutoff]
        
        # Check limit
        if len(self.rate_limit_window) >= self.config.rate_limit_per_minute:
            return False
        
        # Add current request
        self.rate_limit_window.append(now)
        return True
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from open to half-open."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _transition_to_open(self):
        """Transition circuit to open state."""
        self.state = CircuitState.OPEN
        self.state_changed_at = datetime.now()
        self.half_open_calls = 0
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.state_changed_at = datetime.now()
        self.half_open_calls = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
    
    def _transition_to_closed(self):
        """Transition circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.half_open_calls = 0
        self.consecutive_failures = 0
    
    def _trim_call_history(self):
        """Trim call history to sliding window size."""
        if len(self.call_history) > self.config.sliding_window_size:
            self.call_history = self.call_history[-self.config.sliding_window_size:]
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics."""
        with self.lock:
            if not self.call_history:
                return CircuitBreakerMetrics(current_state=self.state, state_changed_at=self.state_changed_at)
            
            total_calls = len(self.call_history)
            successful_calls = sum(1 for call in self.call_history if call.success)
            failed_calls = total_calls - successful_calls
            
            timeouts = sum(1 for call in self.call_history if call.failure_type == FailureType.TIMEOUT)
            rate_limited = sum(1 for call in self.call_history if call.failure_type == FailureType.RATE_LIMIT)
            quality_failures = sum(1 for call in self.call_history if call.failure_type == FailureType.QUALITY_THRESHOLD)
            
            durations = [call.duration for call in self.call_history if call.duration > 0]
            avg_response_time = statistics.mean(durations) if durations else 0.0
            
            success_rate = (successful_calls / total_calls) if total_calls > 0 else 0.0
            
            return CircuitBreakerMetrics(
                total_calls=total_calls,
                successful_calls=successful_calls,
                failed_calls=failed_calls,
                timeouts=timeouts,
                rate_limited=rate_limited,
                quality_failures=quality_failures,
                average_response_time=avg_response_time,
                success_rate=success_rate,
                current_state=self.state,
                state_changed_at=self.state_changed_at,
                last_failure_time=self.last_failure_time,
                consecutive_failures=self.consecutive_failures,
                consecutive_successes=self.consecutive_successes
            )
    
    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.state_changed_at = datetime.now()
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.half_open_calls = 0
            self.last_failure_time = None
    
    def force_open(self):
        """Manually force the circuit breaker to open state."""
        with self.lock:
            self.state = CircuitState.OPEN
            self.state_changed_at = datetime.now()
            self.last_failure_time = datetime.now()
