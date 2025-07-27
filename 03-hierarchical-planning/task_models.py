"""Task models for hierarchical planning."""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel
import uuid


class TaskStatus(Enum):
    """Status of a task in the plan."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """Represents a single task in the hierarchical plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    subtasks: List['Task'] = field(default_factory=list)
    parent_id: Optional[str] = None
    result: Optional[str] = None
    estimated_duration: Optional[int] = None  # in minutes
    actual_duration: Optional[int] = None
    
    def is_ready_to_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task can be executed based on dependencies."""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)
    
    def has_subtasks(self) -> bool:
        """Check if task has subtasks."""
        return len(self.subtasks) > 0
    
    def get_all_subtask_ids(self) -> Set[str]:
        """Get all subtask IDs recursively."""
        subtask_ids = set()
        for subtask in self.subtasks:
            subtask_ids.add(subtask.id)
            subtask_ids.update(subtask.get_all_subtask_ids())
        return subtask_ids


@dataclass 
class ExecutionPlan:
    """Represents the complete hierarchical execution plan."""
    goal: str
    root_tasks: List[Task] = field(default_factory=list)
    task_registry: Dict[str, Task] = field(default_factory=dict)
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    
    def add_task(self, task: Task, parent_id: Optional[str] = None):
        """Add a task to the plan."""
        task.parent_id = parent_id
        self.task_registry[task.id] = task
        
        if parent_id:
            parent_task = self.task_registry.get(parent_id)
            if parent_task:
                parent_task.subtasks.append(task)
        else:
            self.root_tasks.append(task)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to be executed."""
        ready_tasks = []
        
        for task in self.task_registry.values():
            if (task.status == TaskStatus.PENDING and 
                task.is_ready_to_execute(self.completed_tasks) and
                not task.has_subtasks()):
                ready_tasks.append(task)
        
        # Sort by priority
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3
        }
        
        ready_tasks.sort(key=lambda t: priority_order[t.priority])
        return ready_tasks
    
    def mark_task_completed(self, task_id: str, result: str = ""):
        """Mark a task as completed with its result."""
        if task_id in self.task_registry:
            task = self.task_registry[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = result
            self.completed_tasks.add(task_id)
    
    def mark_task_failed(self, task_id: str, error: str = ""):
        """Mark a task as failed."""
        if task_id in self.task_registry:
            task = self.task_registry[task_id]
            task.status = TaskStatus.FAILED
            task.result = f"FAILED: {error}"
            self.failed_tasks.add(task_id)
    
    def is_complete(self) -> bool:
        """Check if all tasks in the plan are completed."""
        for task in self.task_registry.values():
            if not task.has_subtasks() and task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False
        return True
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of plan execution progress."""
        total_leaf_tasks = sum(1 for t in self.task_registry.values() if not t.has_subtasks())
        completed_leaf_tasks = sum(1 for t in self.task_registry.values() 
                                  if not t.has_subtasks() and t.status == TaskStatus.COMPLETED)
        failed_leaf_tasks = sum(1 for t in self.task_registry.values() 
                               if not t.has_subtasks() and t.status == TaskStatus.FAILED)
        
        return {
            "total_tasks": total_leaf_tasks,
            "completed_tasks": completed_leaf_tasks,
            "failed_tasks": failed_leaf_tasks,
            "progress_percentage": (completed_leaf_tasks / total_leaf_tasks * 100) if total_leaf_tasks > 0 else 0,
            "is_complete": self.is_complete()
        }
    
    def get_task_hierarchy_display(self, task: Optional[Task] = None, indent: int = 0) -> str:
        """Get a string representation of the task hierarchy."""
        if task is None:
            result = f"📋 Goal: {self.goal}\n"
            result += "=" * 50 + "\n"
            for root_task in self.root_tasks:
                result += self.get_task_hierarchy_display(root_task, 0)
            return result
        
        # Status icons
        status_icons = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.IN_PROGRESS: "🔄", 
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.BLOCKED: "🚫"
        }
        
        # Priority indicators
        priority_indicators = {
            TaskPriority.CRITICAL: "🔴",
            TaskPriority.HIGH: "🟡",
            TaskPriority.MEDIUM: "🔵",
            TaskPriority.LOW: "⚪"
        }
        
        indent_str = "  " * indent
        result = f"{indent_str}{status_icons.get(task.status, '❓')} {priority_indicators.get(task.priority, '')} {task.title}\n"
        
        if task.description and indent == 0:
            result += f"{indent_str}   📝 {task.description}\n"
        
        if task.dependencies:
            deps = ", ".join(task.dependencies)
            result += f"{indent_str}   🔗 Depends on: {deps}\n"
        
        if task.result and task.status == TaskStatus.COMPLETED:
            result += f"{indent_str}   ✨ Result: {task.result[:100]}{'...' if len(task.result) > 100 else ''}\n"
        
        # Add subtasks
        for subtask in task.subtasks:
            result += self.get_task_hierarchy_display(subtask, indent + 1)
        
        return result


class PlanningRequest(BaseModel):
    """Request for creating a hierarchical plan."""
    goal: str
    max_depth: int = 3
    max_tasks_per_level: int = 5
    context: Optional[str] = None