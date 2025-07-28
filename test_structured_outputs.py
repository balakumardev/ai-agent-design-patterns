#!/usr/bin/env python3
"""
Simple test script to verify structured outputs are working correctly.
"""

import sys
import os

def test_hierarchical_planning_models():
    """Test the hierarchical planning Pydantic models."""
    print("Testing Hierarchical Planning Models...")
    
    try:
        sys.path.append('./03-hierarchical-planning')
        from task_models import TaskDecompositionResponse, TaskModel, SubtaskModel
        from pydantic import ValidationError
        
        # Test data
        sample_data = {
            'tasks': [
                {
                    'title': 'Research topic',
                    'description': 'Conduct thorough research on the topic',
                    'priority': 'high',
                    'estimated_duration': 30,
                    'dependencies': [],
                    'subtasks': [
                        {
                            'title': 'Literature review',
                            'description': 'Review existing literature',
                            'priority': 'medium',
                            'estimated_duration': 20,
                            'dependencies': []
                        }
                    ]
                }
            ]
        }
        
        # Test model creation
        response = TaskDecompositionResponse(**sample_data)
        print("✅ TaskDecompositionResponse model validation passed")
        
        # Test model_dump
        dumped_data = response.model_dump()
        print("✅ model_dump() works correctly")
        
        print(f"   Tasks created: {len(response.tasks)}")
        print(f"   First task title: {response.tasks[0].title}")
        print(f"   Subtasks: {len(response.tasks[0].subtasks)}")
        
    except Exception as e:
        print(f"❌ Hierarchical Planning test failed: {e}")


def test_multi_agent_models():
    """Test the multi-agent coordination Pydantic models."""
    print("\nTesting Multi-Agent Coordination Models...")
    
    try:
        sys.path.append('./04-multi-agent-coordination')
        from agent_models import AgentSetupResponse, AgentModel, AgentCapabilityModel
        
        # Test data
        sample_data = {
            'agents': [
                {
                    'name': 'Researcher Agent',
                    'role': 'researcher',
                    'capabilities': [
                        {
                            'name': 'web_search',
                            'description': 'Search the web for information',
                            'input_types': ['query'],
                            'output_types': ['search_results'],
                            'estimated_duration': 30
                        }
                    ],
                    'specialization': 'Academic research'
                }
            ]
        }
        
        # Test model creation
        response = AgentSetupResponse(**sample_data)
        print("✅ AgentSetupResponse model validation passed")
        
        # Test model_dump
        dumped_data = response.model_dump()
        print("✅ model_dump() works correctly")
        
        print(f"   Agents created: {len(response.agents)}")
        print(f"   First agent name: {response.agents[0].name}")
        print(f"   Capabilities: {len(response.agents[0].capabilities)}")
        
    except Exception as e:
        print(f"❌ Multi-Agent test failed: {e}")


def test_constitutional_ai_models():
    """Test the constitutional AI Pydantic models."""
    print("\nTesting Constitutional AI Models...")
    
    try:
        sys.path.append('./08-constitutional-ai-pattern')
        from constitutional_models import ConstitutionalEvaluationResponse, ConstitutionalViolationModel
        
        # Test data
        sample_data = {
            'violations': [
                {
                    'principle_id': 'safety_harmful_content',
                    'violated': False,
                    'confidence': 0.95,
                    'explanation': 'Content appears safe and harmless',
                    'severity': 'low',
                    'suggested_action': 'log',
                    'modified_content': None
                }
            ],
            'overall_compliant': True,
            'recommended_action': 'log',
            'final_content': 'This is safe content that passed all checks.'
        }
        
        # Test model creation
        response = ConstitutionalEvaluationResponse(**sample_data)
        print("✅ ConstitutionalEvaluationResponse model validation passed")
        
        # Test model_dump
        dumped_data = response.model_dump()
        print("✅ model_dump() works correctly")
        
        print(f"   Violations checked: {len(response.violations)}")
        print(f"   Overall compliant: {response.overall_compliant}")
        print(f"   Recommended action: {response.recommended_action}")
        
    except Exception as e:
        print(f"❌ Constitutional AI test failed: {e}")


def main():
    """Run all tests."""
    print("🧪 Testing Structured Output Models")
    print("=" * 50)
    
    test_hierarchical_planning_models()
    test_multi_agent_models()
    test_constitutional_ai_models()
    
    print("\n" + "=" * 50)
    print("✅ All structured output model tests completed!")


if __name__ == "__main__":
    main()