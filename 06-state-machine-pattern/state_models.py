"""State machine models for predictable agent behavior."""

from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from datetime import datetime


class StateType(Enum):
    """Types of states in the state machine."""
    INITIAL = "initial"
    PROCESSING = "processing"
    WAITING = "waiting"
    DECISION = "decision"
    ACTION = "action"
    FINAL = "final"
    ERROR = "error"


class TransitionCondition(Enum):
    """Common transition conditions."""
    ALWAYS = "always"
    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    ON_TIMEOUT = "on_timeout"
    ON_USER_INPUT = "on_user_input"
    ON_CONDITION = "on_condition"
    ON_EVENT = "on_event"


@dataclass
class StateContext:
    """Context data that flows through the state machine."""
    user_input: str = ""
    current_data: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    error_message: str = ""
    iteration_count: int = 0
    max_iterations: int = 10
    timeout_seconds: int = 300
    started_at: datetime = field(default_factory=datetime.now)
    
    def add_to_history(self, message: str):
        """Add a message to the history."""
        self.history.append(f"{datetime.now().isoformat()}: {message}")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.started_at).total_seconds()
    
    def is_timeout(self) -> bool:
        """Check if the context has timed out."""
        return self.get_elapsed_time() > self.timeout_seconds
    
    def increment_iteration(self):
        """Increment the iteration count."""
        self.iteration_count += 1
    
    def is_max_iterations_reached(self) -> bool:
        """Check if maximum iterations have been reached."""
        return self.iteration_count >= self.max_iterations


@dataclass
class Transition:
    """Represents a transition between states."""
    from_state: str
    to_state: str
    condition: TransitionCondition
    condition_func: Optional[Callable[[StateContext], bool]] = None
    description: str = ""
    
    def can_transition(self, context: StateContext) -> bool:
        """Check if this transition can be taken given the context."""
        if self.condition == TransitionCondition.ALWAYS:
            return True
        elif self.condition == TransitionCondition.ON_SUCCESS:
            return not context.error_message
        elif self.condition == TransitionCondition.ON_FAILURE:
            return bool(context.error_message)
        elif self.condition == TransitionCondition.ON_TIMEOUT:
            return context.is_timeout()
        elif self.condition == TransitionCondition.ON_CONDITION and self.condition_func:
            return self.condition_func(context)
        else:
            return False


class State(ABC):
    """Abstract base class for states."""
    
    def __init__(self, name: str, state_type: StateType, description: str = ""):
        self.name = name
        self.state_type = state_type
        self.description = description
        self.entry_actions: List[Callable[[StateContext], None]] = []
        self.exit_actions: List[Callable[[StateContext], None]] = []
    
    def add_entry_action(self, action: Callable[[StateContext], None]):
        """Add an action to execute when entering this state."""
        self.entry_actions.append(action)
    
    def add_exit_action(self, action: Callable[[StateContext], None]):
        """Add an action to execute when exiting this state."""
        self.exit_actions.append(action)
    
    def on_entry(self, context: StateContext):
        """Execute entry actions."""
        context.add_to_history(f"Entering state: {self.name}")
        for action in self.entry_actions:
            action(context)
    
    def on_exit(self, context: StateContext):
        """Execute exit actions."""
        context.add_to_history(f"Exiting state: {self.name}")
        for action in self.exit_actions:
            action(context)
    
    @abstractmethod
    def execute(self, context: StateContext) -> StateContext:
        """Execute the state's main logic."""
        pass


class InitialState(State):
    """Initial state of the state machine."""
    
    def __init__(self, name: str = "initial", description: str = "Initial state"):
        super().__init__(name, StateType.INITIAL, description)
    
    def execute(self, context: StateContext) -> StateContext:
        """Initialize the context."""
        context.add_to_history("State machine started")
        return context


class FinalState(State):
    """Final state of the state machine."""
    
    def __init__(self, name: str = "final", description: str = "Final state"):
        super().__init__(name, StateType.FINAL, description)
    
    def execute(self, context: StateContext) -> StateContext:
        """Finalize the context."""
        context.add_to_history("State machine completed")
        return context


class ErrorState(State):
    """Error state for handling failures."""
    
    def __init__(self, name: str = "error", description: str = "Error state"):
        super().__init__(name, StateType.ERROR, description)
    
    def execute(self, context: StateContext) -> StateContext:
        """Handle error state."""
        context.add_to_history(f"Error occurred: {context.error_message}")
        return context


class ProcessingState(State):
    """State for processing operations."""
    
    def __init__(self, name: str, description: str = "", processor: Optional[Callable[[StateContext], StateContext]] = None):
        super().__init__(name, StateType.PROCESSING, description)
        self.processor = processor
    
    def execute(self, context: StateContext) -> StateContext:
        """Execute processing logic."""
        if self.processor:
            try:
                return self.processor(context)
            except Exception as e:
                context.error_message = str(e)
                context.add_to_history(f"Processing error in {self.name}: {str(e)}")
        return context


class DecisionState(State):
    """State for making decisions based on context."""
    
    def __init__(self, name: str, description: str = "", decision_func: Optional[Callable[[StateContext], str]] = None):
        super().__init__(name, StateType.DECISION, description)
        self.decision_func = decision_func
    
    def execute(self, context: StateContext) -> StateContext:
        """Execute decision logic."""
        if self.decision_func:
            try:
                decision = self.decision_func(context)
                context.current_data['decision'] = decision
                context.add_to_history(f"Decision made in {self.name}: {decision}")
            except Exception as e:
                context.error_message = str(e)
                context.add_to_history(f"Decision error in {self.name}: {str(e)}")
        return context


class WaitingState(State):
    """State for waiting for external input or events."""
    
    def __init__(self, name: str, description: str = "", wait_condition: Optional[Callable[[StateContext], bool]] = None):
        super().__init__(name, StateType.WAITING, description)
        self.wait_condition = wait_condition
    
    def execute(self, context: StateContext) -> StateContext:
        """Execute waiting logic."""
        context.add_to_history(f"Waiting in state: {self.name}")
        
        if self.wait_condition:
            if self.wait_condition(context):
                context.add_to_history(f"Wait condition satisfied in {self.name}")
            else:
                context.add_to_history(f"Wait condition not satisfied in {self.name}")
        
        return context


@dataclass
class StateMachineDefinition:
    """Definition of a complete state machine."""
    name: str
    description: str
    states: Dict[str, State] = field(default_factory=dict)
    transitions: List[Transition] = field(default_factory=list)
    initial_state: str = "initial"
    final_states: Set[str] = field(default_factory=set)
    error_states: Set[str] = field(default_factory=set)
    
    def add_state(self, state: State):
        """Add a state to the state machine."""
        self.states[state.name] = state
        
        if state.state_type == StateType.FINAL:
            self.final_states.add(state.name)
        elif state.state_type == StateType.ERROR:
            self.error_states.add(state.name)
    
    def add_transition(self, transition: Transition):
        """Add a transition to the state machine."""
        self.transitions.append(transition)
    
    def get_transitions_from_state(self, state_name: str) -> List[Transition]:
        """Get all transitions from a specific state."""
        return [t for t in self.transitions if t.from_state == state_name]
    
    def get_next_state(self, current_state: str, context: StateContext) -> Optional[str]:
        """Get the next state based on current state and context."""
        transitions = self.get_transitions_from_state(current_state)
        
        for transition in transitions:
            if transition.can_transition(context):
                return transition.to_state
        
        return None
    
    def is_final_state(self, state_name: str) -> bool:
        """Check if a state is a final state."""
        return state_name in self.final_states
    
    def is_error_state(self, state_name: str) -> bool:
        """Check if a state is an error state."""
        return state_name in self.error_states
    
    def validate(self) -> List[str]:
        """Validate the state machine definition."""
        errors = []
        
        # Check if initial state exists
        if self.initial_state not in self.states:
            errors.append(f"Initial state '{self.initial_state}' not found in states")
        
        # Check if all transition states exist
        for transition in self.transitions:
            if transition.from_state not in self.states:
                errors.append(f"Transition from unknown state: {transition.from_state}")
            if transition.to_state not in self.states:
                errors.append(f"Transition to unknown state: {transition.to_state}")
        
        # Check if there's at least one final state
        if not self.final_states:
            errors.append("No final states defined")
        
        # Check for unreachable states
        reachable_states = {self.initial_state}
        changed = True
        while changed:
            changed = False
            for transition in self.transitions:
                if transition.from_state in reachable_states and transition.to_state not in reachable_states:
                    reachable_states.add(transition.to_state)
                    changed = True
        
        unreachable_states = set(self.states.keys()) - reachable_states
        if unreachable_states:
            errors.append(f"Unreachable states: {unreachable_states}")
        
        return errors
    
    def to_graphviz(self) -> str:
        """Generate Graphviz DOT representation of the state machine."""
        dot = f'digraph "{self.name}" {{\n'
        dot += '  rankdir=LR;\n'
        dot += '  node [shape=circle];\n'
        
        # Add states
        for state_name, state in self.states.items():
            shape = "circle"
            color = "black"
            
            if state.state_type == StateType.INITIAL:
                shape = "doublecircle"
                color = "green"
            elif state.state_type == StateType.FINAL:
                shape = "doublecircle"
                color = "blue"
            elif state.state_type == StateType.ERROR:
                shape = "doublecircle"
                color = "red"
            elif state.state_type == StateType.DECISION:
                shape = "diamond"
            
            dot += f'  "{state_name}" [shape={shape}, color={color}];\n'
        
        # Add transitions
        for transition in self.transitions:
            label = transition.condition.value
            if transition.description:
                label += f"\\n{transition.description}"
            
            dot += f'  "{transition.from_state}" -> "{transition.to_state}" [label="{label}"];\n'
        
        dot += '}\n'
        return dot
