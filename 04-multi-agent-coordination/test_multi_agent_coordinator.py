"""Tests for the multi-agent coordination system."""

import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import datetime
from agent_models import (
    Agent, AgentRole, AgentCapability, AgentStatus, Message, MessageType,
    CoordinationTask, CoordinationPlan, CoordinationRequest
)
from multi_agent_coordinator import MultiAgentCoordinator


class TestAgentCapability:
    """Test the AgentCapability class."""
    
    def test_capability_creation(self):
        """Test creating an agent capability."""
        capability = AgentCapability(
            name="data_analysis",
            description="Analyze data and generate insights",
            input_types=["csv", "json"],
            output_types=["report", "visualization"],
            estimated_duration=300
        )
        
        assert capability.name == "data_analysis"
        assert capability.description == "Analyze data and generate insights"
        assert capability.input_types == ["csv", "json"]
        assert capability.output_types == ["report", "visualization"]
        assert capability.estimated_duration == 300


class TestAgent:
    """Test the Agent class."""
    
    def test_agent_creation(self):
        """Test creating an agent."""
        capability = AgentCapability(
            name="research",
            description="Conduct research",
            input_types=["query"],
            output_types=["findings"]
        )
        
        agent = Agent(
            name="Research Agent",
            role=AgentRole.RESEARCHER,
            capabilities=[capability]
        )
        
        assert agent.name == "Research Agent"
        assert agent.role == AgentRole.RESEARCHER
        assert len(agent.capabilities) == 1
        assert agent.status == AgentStatus.IDLE
        assert len(agent.id) == 8  # UUID truncated to 8 chars
    
    def test_agent_can_handle_task(self):
        """Test agent task handling capability."""
        capability = AgentCapability(
            name="analysis",
            description="Data analysis",
            input_types=["data", "query"],
            output_types=["report"]
        )
        
        agent = Agent(
            name="Analyst",
            role=AgentRole.ANALYST,
            capabilities=[capability]
        )
        
        assert agent.can_handle_task("data")
        assert agent.can_handle_task("query")
        assert not agent.can_handle_task("video")
    
    def test_agent_get_capability(self):
        """Test getting specific capability."""
        cap1 = AgentCapability(name="research", description="Research capability")
        cap2 = AgentCapability(name="analysis", description="Analysis capability")
        
        agent = Agent(
            name="Multi-skilled Agent",
            capabilities=[cap1, cap2]
        )
        
        assert agent.get_capability("research") == cap1
        assert agent.get_capability("analysis") == cap2
        assert agent.get_capability("nonexistent") is None
    
    def test_agent_message_handling(self):
        """Test agent message handling."""
        agent = Agent(name="Test Agent")
        
        message = Message(
            sender_id="other_agent",
            recipient_id=agent.id,
            content="Hello!"
        )
        
        agent.add_message(message)
        
        assert len(agent.message_history) == 1
        assert agent.message_history[0] == message
        
        recent = agent.get_recent_messages(limit=5)
        assert len(recent) == 1
        assert recent[0] == message


class TestMessage:
    """Test the Message class."""
    
    def test_message_creation(self):
        """Test creating a message."""
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type=MessageType.TASK_REQUEST,
            content="Please analyze this data"
        )
        
        assert message.sender_id == "agent1"
        assert message.recipient_id == "agent2"
        assert message.message_type == MessageType.TASK_REQUEST
        assert message.content == "Please analyze this data"
        assert len(message.id) == 8
        assert isinstance(message.timestamp, datetime)
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            content="Test message"
        )
        
        msg_dict = message.to_dict()
        
        assert msg_dict["sender_id"] == "agent1"
        assert msg_dict["recipient_id"] == "agent2"
        assert msg_dict["content"] == "Test message"
        assert "timestamp" in msg_dict
        assert "id" in msg_dict


class TestCoordinationTask:
    """Test the CoordinationTask class."""
    
    def test_task_creation(self):
        """Test creating a coordination task."""
        task = CoordinationTask(
            title="Data Analysis",
            description="Analyze customer data",
            required_capabilities=["data_analysis", "visualization"]
        )
        
        assert task.title == "Data Analysis"
        assert task.description == "Analyze customer data"
        assert task.required_capabilities == ["data_analysis", "visualization"]
        assert task.status == "pending"
        assert len(task.id) == 8
        assert isinstance(task.created_at, datetime)
    
    def test_task_assignment(self):
        """Test assigning agents to tasks."""
        task = CoordinationTask(title="Test Task")
        
        task.assign_agent("agent1")
        task.assign_agent("agent2")
        task.assign_agent("agent1")  # Duplicate should be ignored
        
        assert len(task.assigned_agents) == 2
        assert "agent1" in task.assigned_agents
        assert "agent2" in task.assigned_agents
    
    def test_task_completion(self):
        """Test task completion status."""
        task = CoordinationTask(title="Test Task")
        
        assert not task.is_complete()
        
        task.status = "completed"
        assert task.is_complete()
        
        task.status = "failed"
        assert task.is_complete()
        
        task.status = "in_progress"
        assert not task.is_complete()


class TestCoordinationPlan:
    """Test the CoordinationPlan class."""
    
    def test_plan_creation(self):
        """Test creating a coordination plan."""
        plan = CoordinationPlan(goal="Test Goal")
        
        assert plan.goal == "Test Goal"
        assert len(plan.tasks) == 0
        assert len(plan.agents) == 0
        assert len(plan.message_queue) == 0
    
    def test_add_agent(self):
        """Test adding agents to the plan."""
        plan = CoordinationPlan(goal="Test Goal")
        agent = Agent(name="Test Agent")
        
        plan.add_agent(agent)
        
        assert agent.id in plan.agents
        assert plan.agents[agent.id] == agent
    
    def test_add_task(self):
        """Test adding tasks to the plan."""
        plan = CoordinationPlan(goal="Test Goal")
        task = CoordinationTask(title="Test Task")
        
        plan.add_task(task)
        
        assert len(plan.tasks) == 1
        assert plan.tasks[0] == task
    
    def test_get_available_agents(self):
        """Test getting available agents with specific capabilities."""
        plan = CoordinationPlan(goal="Test Goal")
        
        # Create agents with different capabilities
        cap1 = AgentCapability(name="research", input_types=["query"])
        cap2 = AgentCapability(name="analysis", input_types=["data"])
        
        agent1 = Agent(name="Researcher", capabilities=[cap1], status=AgentStatus.IDLE)
        agent2 = Agent(name="Analyst", capabilities=[cap2], status=AgentStatus.IDLE)
        agent3 = Agent(name="Busy Agent", capabilities=[cap1], status=AgentStatus.BUSY)
        
        plan.add_agent(agent1)
        plan.add_agent(agent2)
        plan.add_agent(agent3)
        
        # Test getting available agents
        available_researchers = plan.get_available_agents("query")
        available_analysts = plan.get_available_agents("data")
        
        assert len(available_researchers) == 1
        assert available_researchers[0] == agent1
        
        assert len(available_analysts) == 1
        assert available_analysts[0] == agent2
    
    def test_send_message(self):
        """Test sending messages in the coordination system."""
        plan = CoordinationPlan(goal="Test Goal")
        
        agent1 = Agent(name="Agent 1")
        agent2 = Agent(name="Agent 2")
        plan.add_agent(agent1)
        plan.add_agent(agent2)
        
        # Test direct message
        message = Message(
            sender_id=agent1.id,
            recipient_id=agent2.id,
            content="Hello!"
        )
        
        plan.send_message(message)
        
        assert len(plan.message_queue) == 1
        assert len(agent2.message_history) == 1
        assert agent2.message_history[0] == message
        
        # Test broadcast message
        broadcast_message = Message(
            sender_id=agent1.id,
            recipient_id="ALL",
            content="Broadcast message"
        )
        
        plan.send_message(broadcast_message)
        
        assert len(plan.message_queue) == 2
        assert len(agent1.message_history) == 1  # Received broadcast
        assert len(agent2.message_history) == 2  # Received both messages
    
    def test_progress_summary(self):
        """Test getting progress summary."""
        plan = CoordinationPlan(goal="Test Goal")
        
        # Add tasks
        task1 = CoordinationTask(title="Task 1")
        task2 = CoordinationTask(title="Task 2")
        task3 = CoordinationTask(title="Task 3")
        
        plan.add_task(task1)
        plan.add_task(task2)
        plan.add_task(task3)
        
        # Initial progress
        progress = plan.get_progress_summary()
        assert progress["total_tasks"] == 3
        assert progress["completed_tasks"] == 0
        assert progress["failed_tasks"] == 0
        assert progress["in_progress_tasks"] == 0
        assert progress["progress_percentage"] == 0.0
        assert not progress["is_complete"]
        
        # Complete some tasks
        task1.status = "completed"
        plan.completed_tasks.add(task1.id)
        task2.status = "failed"
        plan.failed_tasks.add(task2.id)
        task3.status = "in_progress"
        
        progress = plan.get_progress_summary()
        assert progress["completed_tasks"] == 1
        assert progress["failed_tasks"] == 1
        assert progress["in_progress_tasks"] == 1
        assert progress["progress_percentage"] == 33.33333333333333
        assert not progress["is_complete"]
        
        # Complete all tasks
        task3.status = "completed"
        plan.completed_tasks.add(task3.id)
        
        progress = plan.get_progress_summary()
        assert progress["is_complete"]


class TestMultiAgentCoordinator:
    """Test the MultiAgentCoordinator class."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_coordinator_initialization(self):
        """Test that the coordinator initializes correctly."""
        coordinator = MultiAgentCoordinator()
        
        assert coordinator.model is not None
        assert coordinator.graph is not None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('multi_agent_coordinator.ChatOpenAI')
    def test_coordinate_success(self, mock_chat_openai):
        """Test successful coordination."""
        # Mock the model responses
        setup_response = MagicMock()
        setup_response.content = '''
        {
            "agents": [
                {
                    "name": "Researcher",
                    "role": "researcher",
                    "capabilities": [
                        {
                            "name": "research",
                            "description": "Conduct research",
                            "input_types": ["query"],
                            "output_types": ["findings"],
                            "estimated_duration": 300
                        }
                    ]
                }
            ]
        }
        '''
        
        planning_response = MagicMock()
        planning_response.content = '''
        {
            "tasks": [
                {
                    "title": "Research task",
                    "description": "Conduct initial research",
                    "required_capabilities": ["research"],
                    "assigned_agents": [],
                    "dependencies": []
                }
            ]
        }
        '''
        
        execution_response = MagicMock()
        execution_response.content = "Research completed successfully with detailed findings."
        
        mock_model = MagicMock()
        mock_model.invoke.side_effect = [setup_response, planning_response, execution_response]
        mock_chat_openai.return_value = mock_model
        
        coordinator = MultiAgentCoordinator()
        request = CoordinationRequest(goal="Test coordination goal")
        
        plan = coordinator.coordinate(request)
        
        assert plan.goal == "Test coordination goal"
        assert len(plan.agents) >= 1
        assert len(plan.tasks) >= 1


class TestCoordinationRequest:
    """Test the CoordinationRequest model."""
    
    def test_coordination_request_creation(self):
        """Test creating a coordination request."""
        request = CoordinationRequest(
            goal="Test goal",
            context="Test context",
            required_roles=["researcher", "analyst"],
            max_agents=3,
            timeout_minutes=45
        )
        
        assert request.goal == "Test goal"
        assert request.context == "Test context"
        assert request.required_roles == ["researcher", "analyst"]
        assert request.max_agents == 3
        assert request.timeout_minutes == 45
    
    def test_coordination_request_defaults(self):
        """Test coordination request with default values."""
        request = CoordinationRequest(goal="Test goal")
        
        assert request.goal == "Test goal"
        assert request.context is None
        assert request.required_roles == []
        assert request.max_agents == 5
        assert request.timeout_minutes == 30


if __name__ == "__main__":
    pytest.main([__file__])
