"""Tests for the state machine agent."""

import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from state_models import (
    StateContext, StateMachineDefinition, Transition, TransitionCondition,
    StateType, InitialState, FinalState, ErrorState, ProcessingState, 
    DecisionState, WaitingState
)
from state_machine_agent import StateMachineAgent, ExecutionResult


class TestStateContext:
    """Test the StateContext class."""
    
    def test_context_creation(self):
        """Test creating a state context."""
        context = StateContext(
            user_input="test input",
            max_iterations=5,
            timeout_seconds=60
        )
        
        assert context.user_input == "test input"
        assert context.max_iterations == 5
        assert context.timeout_seconds == 60
        assert context.iteration_count == 0
        assert len(context.history) == 0
    
    def test_add_to_history(self):
        """Test adding messages to history."""
        context = StateContext()
        
        context.add_to_history("Test message")
        
        assert len(context.history) == 1
        assert "Test message" in context.history[0]
    
    def test_iteration_management(self):
        """Test iteration counting."""
        context = StateContext(max_iterations=3)
        
        assert not context.is_max_iterations_reached()
        
        context.increment_iteration()
        assert context.iteration_count == 1
        assert not context.is_max_iterations_reached()
        
        context.increment_iteration()
        context.increment_iteration()
        assert context.iteration_count == 3
        assert context.is_max_iterations_reached()
    
    def test_timeout_management(self):
        """Test timeout functionality."""
        # Create context with very short timeout
        context = StateContext(timeout_seconds=0.1)
        
        # Initially should not be timed out
        assert not context.is_timeout()
        
        # Wait a bit and check again
        import time
        time.sleep(0.2)
        assert context.is_timeout()
    
    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        context = StateContext()
        
        # Should have some elapsed time
        elapsed = context.get_elapsed_time()
        assert elapsed >= 0
        assert elapsed < 1  # Should be very small for new context


class TestTransition:
    """Test the Transition class."""
    
    def test_transition_creation(self):
        """Test creating a transition."""
        transition = Transition(
            from_state="state1",
            to_state="state2",
            condition=TransitionCondition.ON_SUCCESS,
            description="Test transition"
        )
        
        assert transition.from_state == "state1"
        assert transition.to_state == "state2"
        assert transition.condition == TransitionCondition.ON_SUCCESS
        assert transition.description == "Test transition"
    
    def test_always_condition(self):
        """Test ALWAYS transition condition."""
        transition = Transition("state1", "state2", TransitionCondition.ALWAYS)
        context = StateContext()
        
        assert transition.can_transition(context)
    
    def test_success_condition(self):
        """Test ON_SUCCESS transition condition."""
        transition = Transition("state1", "state2", TransitionCondition.ON_SUCCESS)
        
        # Should transition when no error
        context = StateContext()
        assert transition.can_transition(context)
        
        # Should not transition when there's an error
        context.error_message = "Some error"
        assert not transition.can_transition(context)
    
    def test_failure_condition(self):
        """Test ON_FAILURE transition condition."""
        transition = Transition("state1", "state2", TransitionCondition.ON_FAILURE)
        
        # Should not transition when no error
        context = StateContext()
        assert not transition.can_transition(context)
        
        # Should transition when there's an error
        context.error_message = "Some error"
        assert transition.can_transition(context)
    
    def test_custom_condition(self):
        """Test custom condition function."""
        def custom_condition(ctx):
            return ctx.current_data.get('test_flag', False)
        
        transition = Transition(
            "state1", "state2", 
            TransitionCondition.ON_CONDITION,
            condition_func=custom_condition
        )
        
        # Should not transition when flag is False
        context = StateContext()
        assert not transition.can_transition(context)
        
        # Should transition when flag is True
        context.current_data['test_flag'] = True
        assert transition.can_transition(context)


class TestStates:
    """Test various state types."""
    
    def test_initial_state(self):
        """Test initial state."""
        state = InitialState("start", "Starting state")
        context = StateContext()
        
        result = state.execute(context)
        
        assert result == context
        assert len(context.history) > 0
    
    def test_final_state(self):
        """Test final state."""
        state = FinalState("end", "Ending state")
        context = StateContext()
        
        result = state.execute(context)
        
        assert result == context
        assert len(context.history) > 0
    
    def test_error_state(self):
        """Test error state."""
        state = ErrorState("error", "Error state")
        context = StateContext(error_message="Test error")
        
        result = state.execute(context)
        
        assert result == context
        assert len(context.history) > 0
    
    def test_processing_state(self):
        """Test processing state."""
        def test_processor(ctx):
            ctx.current_data['processed'] = True
            return ctx
        
        state = ProcessingState("process", "Processing state", test_processor)
        context = StateContext()
        
        result = state.execute(context)
        
        assert result == context
        assert context.current_data['processed'] is True
    
    def test_processing_state_error(self):
        """Test processing state with error."""
        def error_processor(ctx):
            raise ValueError("Processing error")
        
        state = ProcessingState("process", "Processing state", error_processor)
        context = StateContext()
        
        result = state.execute(context)
        
        assert result == context
        assert "Processing error" in context.error_message
    
    def test_decision_state(self):
        """Test decision state."""
        def test_decision(ctx):
            return "option_a"
        
        state = DecisionState("decide", "Decision state", test_decision)
        context = StateContext()
        
        result = state.execute(context)
        
        assert result == context
        assert context.current_data['decision'] == "option_a"
    
    def test_waiting_state(self):
        """Test waiting state."""
        def wait_condition(ctx):
            return ctx.current_data.get('ready', False)
        
        state = WaitingState("wait", "Waiting state", wait_condition)
        context = StateContext()
        
        result = state.execute(context)
        
        assert result == context
        assert len(context.history) > 0


class TestStateMachineDefinition:
    """Test the StateMachineDefinition class."""
    
    def test_definition_creation(self):
        """Test creating a state machine definition."""
        definition = StateMachineDefinition(
            name="test_machine",
            description="Test state machine",
            initial_state="start"
        )
        
        assert definition.name == "test_machine"
        assert definition.description == "Test state machine"
        assert definition.initial_state == "start"
        assert len(definition.states) == 0
        assert len(definition.transitions) == 0
    
    def test_add_states(self):
        """Test adding states to definition."""
        definition = StateMachineDefinition("test", "Test")
        
        initial = InitialState("start")
        final = FinalState("end")
        error = ErrorState("error")
        
        definition.add_state(initial)
        definition.add_state(final)
        definition.add_state(error)
        
        assert len(definition.states) == 3
        assert "start" in definition.states
        assert "end" in definition.final_states
        assert "error" in definition.error_states
    
    def test_add_transitions(self):
        """Test adding transitions to definition."""
        definition = StateMachineDefinition("test", "Test")
        
        transition = Transition("start", "end", TransitionCondition.ALWAYS)
        definition.add_transition(transition)
        
        assert len(definition.transitions) == 1
        assert definition.transitions[0] == transition
    
    def test_get_transitions_from_state(self):
        """Test getting transitions from a specific state."""
        definition = StateMachineDefinition("test", "Test")
        
        t1 = Transition("start", "middle", TransitionCondition.ALWAYS)
        t2 = Transition("start", "end", TransitionCondition.ON_FAILURE)
        t3 = Transition("middle", "end", TransitionCondition.ALWAYS)
        
        definition.add_transition(t1)
        definition.add_transition(t2)
        definition.add_transition(t3)
        
        start_transitions = definition.get_transitions_from_state("start")
        assert len(start_transitions) == 2
        assert t1 in start_transitions
        assert t2 in start_transitions
        
        middle_transitions = definition.get_transitions_from_state("middle")
        assert len(middle_transitions) == 1
        assert t3 in middle_transitions
    
    def test_get_next_state(self):
        """Test getting next state based on context."""
        definition = StateMachineDefinition("test", "Test")
        
        # Add transitions
        t1 = Transition("start", "success", TransitionCondition.ON_SUCCESS)
        t2 = Transition("start", "error", TransitionCondition.ON_FAILURE)
        
        definition.add_transition(t1)
        definition.add_transition(t2)
        
        # Test success path
        context = StateContext()
        next_state = definition.get_next_state("start", context)
        assert next_state == "success"
        
        # Test failure path
        context.error_message = "Error occurred"
        next_state = definition.get_next_state("start", context)
        assert next_state == "error"
    
    def test_validation(self):
        """Test state machine validation."""
        definition = StateMachineDefinition("test", "Test", initial_state="start")
        
        # Should have errors initially
        errors = definition.validate()
        assert len(errors) > 0
        
        # Add required states
        definition.add_state(InitialState("start"))
        definition.add_state(FinalState("end"))
        definition.add_transition(Transition("start", "end", TransitionCondition.ALWAYS))
        
        # Should be valid now
        errors = definition.validate()
        assert len(errors) == 0
    
    def test_graphviz_generation(self):
        """Test Graphviz DOT generation."""
        definition = StateMachineDefinition("test", "Test")
        
        definition.add_state(InitialState("start"))
        definition.add_state(FinalState("end"))
        definition.add_transition(Transition("start", "end", TransitionCondition.ALWAYS))
        
        dot = definition.to_graphviz()
        
        assert 'digraph "test"' in dot
        assert '"start"' in dot
        assert '"end"' in dot
        assert 'start" -> "end"' in dot


class TestStateMachineAgent:
    """Test the StateMachineAgent class."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = StateMachineAgent()
        
        assert agent.model is not None
        assert len(agent.state_machines) == 0
        assert agent.current_execution is None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_register_state_machine(self):
        """Test registering a state machine."""
        agent = StateMachineAgent()
        
        # Create valid state machine
        definition = StateMachineDefinition("test", "Test")
        definition.add_state(InitialState("start"))
        definition.add_state(FinalState("end"))
        definition.add_transition(Transition("start", "end", TransitionCondition.ALWAYS))
        
        agent.register_state_machine(definition)
        
        assert "test" in agent.state_machines
        assert agent.state_machines["test"] == definition
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_register_invalid_state_machine(self):
        """Test registering an invalid state machine."""
        agent = StateMachineAgent()
        
        # Create invalid state machine (no final states)
        definition = StateMachineDefinition("test", "Test")
        definition.add_state(InitialState("start"))
        
        with pytest.raises(ValueError):
            agent.register_state_machine(definition)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_execute_simple_state_machine(self):
        """Test executing a simple state machine."""
        agent = StateMachineAgent()
        
        # Create simple state machine
        definition = StateMachineDefinition("test", "Test")
        definition.add_state(InitialState("start"))
        definition.add_state(FinalState("end"))
        definition.add_transition(Transition("start", "end", TransitionCondition.ALWAYS))
        
        agent.register_state_machine(definition)
        
        # Execute
        context = StateContext(user_input="test input")
        result = agent.execute_state_machine("test", context, trace_execution=False)
        
        assert isinstance(result, ExecutionResult)
        assert result.success
        assert result.final_state == "end"
        assert len(result.execution_path) == 2  # start -> end
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_execute_nonexistent_state_machine(self):
        """Test executing a non-existent state machine."""
        agent = StateMachineAgent()
        
        with pytest.raises(ValueError):
            agent.execute_state_machine("nonexistent", StateContext())
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_get_state_machine_info(self):
        """Test getting state machine information."""
        agent = StateMachineAgent()
        
        # Create state machine
        definition = StateMachineDefinition("test", "Test machine")
        definition.add_state(InitialState("start"))
        definition.add_state(FinalState("end"))
        definition.add_transition(Transition("start", "end", TransitionCondition.ALWAYS))
        
        agent.register_state_machine(definition)
        
        # Get info
        info = agent.get_state_machine_info("test")
        
        assert info["name"] == "test"
        assert info["description"] == "Test machine"
        assert len(info["states"]) == 2
        assert len(info["transitions"]) == 1
        assert info["initial_state"] == "start"
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_get_nonexistent_state_machine_info(self):
        """Test getting info for non-existent state machine."""
        agent = StateMachineAgent()
        
        info = agent.get_state_machine_info("nonexistent")
        assert info == {}


class TestExecutionResult:
    """Test the ExecutionResult class."""
    
    def test_execution_result_creation(self):
        """Test creating an execution result."""
        context = StateContext(user_input="test")
        execution_path = ["start", "middle", "end"]
        
        result = ExecutionResult(
            final_state="end",
            context=context,
            execution_path=execution_path,
            success=True
        )
        
        assert result.final_state == "end"
        assert result.context == context
        assert result.execution_path == execution_path
        assert result.success is True
        assert result.error_message == ""


if __name__ == "__main__":
    pytest.main([__file__])
