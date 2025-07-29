"""Tests for the hierarchical planning agent."""

import pytest
import os
from unittest.mock import patch, MagicMock
from task_models import Task, TaskStatus, TaskPriority, ExecutionPlan, PlanningRequest
from hierarchical_planner import HierarchicalPlanner


class TestTask:
    """Test the Task class."""
    
    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            title="Test Task",
            description="A test task",
            priority=TaskPriority.HIGH,
            estimated_duration=30
        )
        
        assert task.title == "Test Task"
        assert task.description == "A test task"
        assert task.priority == TaskPriority.HIGH
        assert task.estimated_duration == 30
        assert task.status == TaskStatus.PENDING
        assert len(task.id) == 8  # UUID truncated to 8 chars
    
    def test_task_dependencies(self):
        """Test task dependency checking."""
        task = Task(
            title="Dependent Task",
            dependencies=["task1", "task2"]
        )
        
        # Not ready when dependencies not completed
        assert not task.is_ready_to_execute(set())
        assert not task.is_ready_to_execute({"task1"})
        
        # Ready when all dependencies completed
        assert task.is_ready_to_execute({"task1", "task2"})
        assert task.is_ready_to_execute({"task1", "task2", "task3"})
    
    def test_task_subtasks(self):
        """Test task subtask functionality."""
        parent_task = Task(title="Parent Task")
        child_task = Task(title="Child Task")
        
        assert not parent_task.has_subtasks()
        
        parent_task.subtasks.append(child_task)
        assert parent_task.has_subtasks()
        
        subtask_ids = parent_task.get_all_subtask_ids()
        assert child_task.id in subtask_ids


class TestExecutionPlan:
    """Test the ExecutionPlan class."""
    
    def test_plan_creation(self):
        """Test creating an execution plan."""
        plan = ExecutionPlan(goal="Test Goal")
        
        assert plan.goal == "Test Goal"
        assert len(plan.root_tasks) == 0
        assert len(plan.task_registry) == 0
        assert len(plan.completed_tasks) == 0
    
    def test_add_task(self):
        """Test adding tasks to the plan."""
        plan = ExecutionPlan(goal="Test Goal")
        
        # Add root task
        root_task = Task(title="Root Task")
        plan.add_task(root_task)
        
        assert len(plan.root_tasks) == 1
        assert root_task.id in plan.task_registry
        assert plan.root_tasks[0] == root_task
        
        # Add child task
        child_task = Task(title="Child Task")
        plan.add_task(child_task, parent_id=root_task.id)
        
        assert len(plan.root_tasks) == 1  # Still only one root task
        assert child_task.id in plan.task_registry
        assert child_task in root_task.subtasks
        assert child_task.parent_id == root_task.id
    
    def test_get_ready_tasks(self):
        """Test getting ready tasks."""
        plan = ExecutionPlan(goal="Test Goal")
        
        # Create tasks with dependencies
        task1 = Task(title="Task 1", priority=TaskPriority.HIGH)
        task2 = Task(title="Task 2", dependencies=[task1.id], priority=TaskPriority.MEDIUM)
        task3 = Task(title="Task 3", priority=TaskPriority.CRITICAL)
        
        plan.add_task(task1)
        plan.add_task(task2)
        plan.add_task(task3)
        
        # Initially, only tasks without dependencies should be ready
        ready_tasks = plan.get_ready_tasks()
        ready_titles = [t.title for t in ready_tasks]
        
        assert "Task 1" in ready_titles
        assert "Task 3" in ready_titles
        assert "Task 2" not in ready_titles
        
        # Tasks should be sorted by priority (CRITICAL first)
        assert ready_tasks[0].title == "Task 3"
        assert ready_tasks[1].title == "Task 1"
        
        # After completing task1, task2 should become ready
        plan.mark_task_completed(task1.id, "Task 1 completed")
        ready_tasks = plan.get_ready_tasks()
        ready_titles = [t.title for t in ready_tasks]
        
        assert "Task 2" in ready_titles
        assert "Task 1" not in ready_titles  # Already completed
    
    def test_task_completion(self):
        """Test marking tasks as completed."""
        plan = ExecutionPlan(goal="Test Goal")
        task = Task(title="Test Task")
        plan.add_task(task)
        
        # Mark as completed
        plan.mark_task_completed(task.id, "Task completed successfully")
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Task completed successfully"
        assert task.id in plan.completed_tasks
    
    def test_task_failure(self):
        """Test marking tasks as failed."""
        plan = ExecutionPlan(goal="Test Goal")
        task = Task(title="Test Task")
        plan.add_task(task)
        
        # Mark as failed
        plan.mark_task_failed(task.id, "Task failed due to error")
        
        assert task.status == TaskStatus.FAILED
        assert "FAILED: Task failed due to error" in task.result
        assert task.id in plan.failed_tasks
    
    def test_plan_completion(self):
        """Test checking if plan is complete."""
        plan = ExecutionPlan(goal="Test Goal")
        
        # Empty plan should be complete
        assert plan.is_complete()
        
        # Add tasks
        task1 = Task(title="Task 1")
        task2 = Task(title="Task 2")
        plan.add_task(task1)
        plan.add_task(task2)
        
        # Plan with pending tasks should not be complete
        assert not plan.is_complete()
        
        # Complete one task
        plan.mark_task_completed(task1.id, "Done")
        assert not plan.is_complete()
        
        # Complete all tasks
        plan.mark_task_completed(task2.id, "Done")
        assert plan.is_complete()
    
    def test_progress_summary(self):
        """Test getting progress summary."""
        plan = ExecutionPlan(goal="Test Goal")
        
        # Add tasks
        task1 = Task(title="Task 1")
        task2 = Task(title="Task 2")
        task3 = Task(title="Task 3")
        
        plan.add_task(task1)
        plan.add_task(task2)
        plan.add_task(task3)
        
        # Initial progress
        progress = plan.get_progress_summary()
        assert progress["total_tasks"] == 3
        assert progress["completed_tasks"] == 0
        assert progress["failed_tasks"] == 0
        assert progress["progress_percentage"] == 0.0
        assert not progress["is_complete"]
        
        # Complete some tasks
        plan.mark_task_completed(task1.id, "Done")
        plan.mark_task_failed(task2.id, "Error")
        
        progress = plan.get_progress_summary()
        assert progress["completed_tasks"] == 1
        assert progress["failed_tasks"] == 1
        assert progress["progress_percentage"] == 33.33333333333333
        assert not progress["is_complete"]
        
        # Complete all tasks
        plan.mark_task_completed(task3.id, "Done")
        
        progress = plan.get_progress_summary()
        assert progress["completed_tasks"] == 2
        assert progress["failed_tasks"] == 1
        assert progress["is_complete"]


class TestHierarchicalPlanner:
    """Test the HierarchicalPlanner class."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_planner_initialization(self):
        """Test that the planner initializes correctly."""
        planner = HierarchicalPlanner()
        
        assert planner.model is not None
        assert planner.graph is not None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_create_plan_success(self):
        """Test successful plan creation."""
        # Import the structured response models
        from task_models import TaskDecompositionResponse, TaskModel, SubtaskModel
        
        # Create actual structured response objects
        subtask = SubtaskModel(
            title="Literature review",
            description="Review existing literature", 
            priority="medium",
            estimated_duration=30,
            dependencies=[]
        )
        
        task = TaskModel(
            title="Research phase",
            description="Conduct initial research",
            priority="high", 
            estimated_duration=60,
            dependencies=[],
            subtasks=[subtask]
        )
        
        structured_response = TaskDecompositionResponse(tasks=[task])
        
        # Mock the decompose_goal method directly
        planner = HierarchicalPlanner()
        
        # Create expected execution plan manually
        execution_plan = ExecutionPlan(goal="Test goal")
        
        # Mock the decompose_goal method to return our expected structure  
        with patch.object(planner, 'decompose_goal') as mock_decompose:
            mock_decompose.return_value = {"execution_plan": execution_plan}
            
            # Manually add our expected tasks to the plan
            from task_models import Task, TaskPriority
            research_task = Task(
                title="Research phase",
                description="Conduct initial research",
                priority=TaskPriority.HIGH,
                estimated_duration=60
            )
            
            lit_review_task = Task(
                title="Literature review", 
                description="Review existing literature",
                priority=TaskPriority.MEDIUM,
                estimated_duration=30
            )
            
            execution_plan.add_task(research_task)
            execution_plan.add_task(lit_review_task, research_task.id)
            
            request = PlanningRequest(goal="Test goal", max_depth=2)
            plan = planner.create_plan(request)
            
            assert plan.goal == "Test goal"
            assert len(plan.root_tasks) == 1
            assert plan.root_tasks[0].title == "Research phase"
            assert len(plan.root_tasks[0].subtasks) == 1
            assert plan.root_tasks[0].subtasks[0].title == "Literature review"
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('shared_utils.create_llm')
    def test_create_plan_error_handling(self, mock_create_llm):
        """Test plan creation with error handling."""
        # Mock the model to raise an exception
        mock_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.side_effect = Exception("API Error")
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_create_llm.return_value = mock_model
        
        planner = HierarchicalPlanner()
        request = PlanningRequest(goal="Test goal")
        
        plan = planner.create_plan(request)
        
        # Should return a fallback plan
        assert plan.goal == "Test goal"
        assert len(plan.root_tasks) >= 1  # Should have at least a fallback task
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_execute_plan(self):
        """Test plan execution."""
        planner = HierarchicalPlanner()
        
        # Create a simple plan
        plan = ExecutionPlan(goal="Test goal")
        task = Task(title="Test task", description="A simple test task")
        plan.add_task(task)
        
        # Mock the graph.invoke method to return a completed plan
        with patch.object(planner.graph, 'invoke') as mock_invoke:
            # Create a completed execution plan for the mock
            completed_plan = ExecutionPlan(goal="Test goal")
            completed_task = Task(title="Test task", description="A simple test task")
            completed_plan.add_task(completed_task)
            completed_plan.mark_task_completed(completed_task.id, "Task completed successfully with detailed results.")
            
            mock_invoke.return_value = {"execution_plan": completed_plan}
            
            # Execute the plan
            executed_plan = planner.execute_plan(plan)
            
            # Check that the method returns an execution plan
            assert isinstance(executed_plan, ExecutionPlan)
            assert executed_plan.goal == "Test goal"


class TestPlanningRequest:
    """Test the PlanningRequest model."""
    
    def test_planning_request_creation(self):
        """Test creating a planning request."""
        request = PlanningRequest(
            goal="Test goal",
            max_depth=3,
            max_tasks_per_level=5,
            context="Test context"
        )
        
        assert request.goal == "Test goal"
        assert request.max_depth == 3
        assert request.max_tasks_per_level == 5
        assert request.context == "Test context"
    
    def test_planning_request_defaults(self):
        """Test planning request with default values."""
        request = PlanningRequest(goal="Test goal")
        
        assert request.goal == "Test goal"
        assert request.max_depth == 3
        assert request.max_tasks_per_level == 5
        assert request.context is None


if __name__ == "__main__":
    pytest.main([__file__])
