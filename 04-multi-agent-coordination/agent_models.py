"""Agent models for multi-agent coordination."""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


class AgentRole(Enum):
    """Roles that agents can play in the coordination system."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"
    EXECUTOR = "executor"
    SPECIALIST = "specialist"


class MessageType(Enum):
    """Types of messages agents can send."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    INFORMATION_SHARE = "information_share"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"


class AgentStatus(Enum):
    """Status of an agent."""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class Message:
    """Represents a message between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    recipient_id: str = ""  # Can be "ALL" for broadcast
    message_type: MessageType = MessageType.INFORMATION_SHARE
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AgentCapability:
    """Represents a capability that an agent has."""
    name: str
    description: str
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    estimated_duration: Optional[int] = None  # in seconds


@dataclass
class Agent:
    """Represents an agent in the multi-agent system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    role: AgentRole = AgentRole.EXECUTOR
    capabilities: List[AgentCapability] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    message_history: List[Message] = field(default_factory=list)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    
    def can_handle_task(self, task_type: str) -> bool:
        """Check if agent can handle a specific task type."""
        return any(task_type in cap.input_types for cap in self.capabilities)
    
    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """Get a specific capability by name."""
        return next((cap for cap in self.capabilities if cap.name == name), None)
    
    def add_message(self, message: Message):
        """Add a message to the agent's history."""
        self.message_history.append(message)
    
    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get recent messages."""
        return self.message_history[-limit:]


@dataclass
class CoordinationTask:
    """Represents a task that requires coordination between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    assigned_agents: List[str] = field(default_factory=list)  # Agent IDs
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.status in ["completed", "failed"]
    
    def assign_agent(self, agent_id: str):
        """Assign an agent to this task."""
        if agent_id not in self.assigned_agents:
            self.assigned_agents.append(agent_id)


@dataclass
class CoordinationPlan:
    """Represents a plan for coordinating multiple agents."""
    goal: str
    tasks: List[CoordinationTask] = field(default_factory=list)
    agents: Dict[str, Agent] = field(default_factory=dict)
    message_queue: List[Message] = field(default_factory=list)
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    
    def add_agent(self, agent: Agent):
        """Add an agent to the coordination plan."""
        self.agents[agent.id] = agent
    
    def add_task(self, task: CoordinationTask):
        """Add a task to the coordination plan."""
        self.tasks.append(task)
    
    def get_available_agents(self, capability: str) -> List[Agent]:
        """Get agents that have a specific capability and are available."""
        available = []
        for agent in self.agents.values():
            if (agent.status == AgentStatus.IDLE and 
                agent.can_handle_task(capability)):
                available.append(agent)
        return available
    
    def send_message(self, message: Message):
        """Send a message in the coordination system."""
        self.message_queue.append(message)
        
        # Add to recipient's history
        if message.recipient_id in self.agents:
            self.agents[message.recipient_id].add_message(message)
        elif message.recipient_id == "ALL":
            # Broadcast to all agents
            for agent in self.agents.values():
                agent.add_message(message)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of coordination progress."""
        total_tasks = len(self.tasks)
        completed_tasks = len(self.completed_tasks)
        failed_tasks = len(self.failed_tasks)
        in_progress_tasks = sum(1 for task in self.tasks if task.status == "in_progress")
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "progress_percentage": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "is_complete": completed_tasks + failed_tasks == total_tasks
        }
    
    def get_agent_status_summary(self) -> Dict[str, int]:
        """Get a summary of agent statuses."""
        status_counts = {}
        for agent in self.agents.values():
            status = agent.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts


# Pydantic models for structured outputs
class AgentCapabilityModel(BaseModel):
    """Pydantic model for agent capabilities."""
    name: str
    description: str
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    estimated_duration: Optional[int] = Field(None, description="Duration in seconds")


class AgentModel(BaseModel):
    """Pydantic model for agents in structured output."""
    name: str
    role: str = Field(..., pattern=r"^(coordinator|researcher|analyst|writer|reviewer|executor|specialist)$")
    capabilities: List[AgentCapabilityModel]
    specialization: Optional[str] = None


class AgentSetupResponse(BaseModel):
    """Structured response for agent setup."""
    agents: List[AgentModel]


class CoordinationTaskModel(BaseModel):
    """Pydantic model for coordination tasks."""
    title: str
    description: str
    required_capabilities: List[str]
    priority: str = Field(..., pattern=r"^(high|medium|low)$")
    estimated_duration: Optional[int] = Field(None, description="Duration in minutes")
    dependencies: List[str] = Field(default_factory=list)


class CoordinationPlanResponse(BaseModel):
    """Structured response for coordination planning."""
    tasks: List[CoordinationTaskModel]
    execution_order: List[str] = Field(default_factory=list)


class CoordinationRequest(BaseModel):
    """Request for multi-agent coordination."""
    goal: str
    context: Optional[str] = None
    required_roles: List[str] = []
    max_agents: int = 5
    timeout_minutes: int = 30
