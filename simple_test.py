#!/usr/bin/env python3
"""Simple test to verify imports work."""

print("Testing imports...")

try:
    import sys
    sys.path.append('./03-hierarchical-planning')
    from task_models import TaskDecompositionResponse
    print("✅ Hierarchical planning imports work")
except Exception as e:
    print(f"❌ Hierarchical planning import failed: {e}")

try:
    sys.path.append('./04-multi-agent-coordination') 
    from agent_models import AgentSetupResponse
    print("✅ Multi-agent coordination imports work")
except Exception as e:
    print(f"❌ Multi-agent coordination import failed: {e}")

try:
    sys.path.append('./08-constitutional-ai-pattern')
    from constitutional_models import ConstitutionalEvaluationResponse
    print("✅ Constitutional AI imports work")
except Exception as e:
    print(f"❌ Constitutional AI import failed: {e}")

print("Import tests completed")