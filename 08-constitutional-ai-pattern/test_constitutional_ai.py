"""Tests for the constitutional AI implementation."""

import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime
from constitutional_models import (
    ConstitutionalPrinciple, ConstitutionalAssessment, ViolationResult,
    ConstitutionalPrincipleLibrary, PrincipleType, ViolationSeverity, ActionType
)
from constitutional_agent import (
    ConstitutionalAI, ConstitutionalResponse, LLMConstitutionalValidator,
    LLMConstitutionalModifier
)


class TestConstitutionalPrinciple:
    """Test the ConstitutionalPrinciple class."""
    
    def test_principle_creation(self):
        """Test creating a constitutional principle."""
        principle = ConstitutionalPrinciple(
            id="test_principle",
            name="Test Principle",
            description="A test principle",
            principle_type=PrincipleType.SAFETY,
            severity=ViolationSeverity.HIGH,
            action=ActionType.MODIFY,
            prompt_template="Test prompt: {content}",
            examples=[{"test": "example"}],
            enabled=True,
            weight=1.5
        )
        
        assert principle.id == "test_principle"
        assert principle.name == "Test Principle"
        assert principle.principle_type == PrincipleType.SAFETY
        assert principle.severity == ViolationSeverity.HIGH
        assert principle.action == ActionType.MODIFY
        assert principle.enabled is True
        assert principle.weight == 1.5
    
    def test_principle_to_dict(self):
        """Test converting principle to dictionary."""
        principle = ConstitutionalPrinciple(
            id="test",
            name="Test",
            description="Test principle",
            principle_type=PrincipleType.ETHICS,
            severity=ViolationSeverity.MEDIUM,
            action=ActionType.WARN,
            prompt_template="Test: {content}"
        )
        
        data = principle.to_dict()
        
        assert data["id"] == "test"
        assert data["name"] == "Test"
        assert data["principle_type"] == "ethics"
        assert data["severity"] == "medium"
        assert data["action"] == "warn"
    
    def test_principle_from_dict(self):
        """Test creating principle from dictionary."""
        data = {
            "id": "test",
            "name": "Test",
            "description": "Test principle",
            "principle_type": "privacy",
            "severity": "low",
            "action": "log",
            "prompt_template": "Test: {content}",
            "examples": [],
            "enabled": False,
            "weight": 0.5
        }
        
        principle = ConstitutionalPrinciple.from_dict(data)
        
        assert principle.id == "test"
        assert principle.principle_type == PrincipleType.PRIVACY
        assert principle.severity == ViolationSeverity.LOW
        assert principle.action == ActionType.LOG
        assert principle.enabled is False
        assert principle.weight == 0.5


class TestViolationResult:
    """Test the ViolationResult class."""
    
    def test_violation_result_creation(self):
        """Test creating a violation result."""
        result = ViolationResult(
            principle_id="test_principle",
            violated=True,
            confidence=0.85,
            explanation="Test violation",
            severity=ViolationSeverity.HIGH,
            suggested_action=ActionType.MODIFY,
            modified_content="Modified content"
        )
        
        assert result.principle_id == "test_principle"
        assert result.violated is True
        assert result.confidence == 0.85
        assert result.explanation == "Test violation"
        assert result.severity == ViolationSeverity.HIGH
        assert result.suggested_action == ActionType.MODIFY
        assert result.modified_content == "Modified content"
    
    def test_violation_result_to_dict(self):
        """Test converting violation result to dictionary."""
        result = ViolationResult(
            principle_id="test",
            violated=False,
            confidence=0.9,
            explanation="No violation",
            severity=ViolationSeverity.LOW,
            suggested_action=ActionType.LOG
        )
        
        data = result.to_dict()
        
        assert data["principle_id"] == "test"
        assert data["violated"] is False
        assert data["confidence"] == 0.9
        assert data["severity"] == "low"
        assert data["suggested_action"] == "log"


class TestConstitutionalAssessment:
    """Test the ConstitutionalAssessment class."""
    
    def test_assessment_creation(self):
        """Test creating a constitutional assessment."""
        violations = [
            ViolationResult("test1", True, 0.8, "Violation 1", ViolationSeverity.HIGH, ActionType.MODIFY),
            ViolationResult("test2", False, 0.9, "No violation", ViolationSeverity.LOW, ActionType.LOG)
        ]
        
        assessment = ConstitutionalAssessment(
            content="Test content",
            violations=violations,
            overall_score=0.7,
            is_compliant=False,
            recommended_action=ActionType.MODIFY,
            final_content="Modified content"
        )
        
        assert assessment.content == "Test content"
        assert len(assessment.violations) == 2
        assert assessment.overall_score == 0.7
        assert assessment.is_compliant is False
        assert assessment.recommended_action == ActionType.MODIFY
        assert assessment.final_content == "Modified content"
    
    def test_get_violations_by_severity(self):
        """Test getting violations by severity."""
        violations = [
            ViolationResult("test1", True, 0.8, "High violation", ViolationSeverity.HIGH, ActionType.MODIFY),
            ViolationResult("test2", True, 0.7, "Medium violation", ViolationSeverity.MEDIUM, ActionType.WARN),
            ViolationResult("test3", False, 0.9, "No violation", ViolationSeverity.LOW, ActionType.LOG)
        ]
        
        assessment = ConstitutionalAssessment(
            content="Test",
            violations=violations,
            overall_score=0.5,
            is_compliant=False,
            recommended_action=ActionType.MODIFY,
            final_content="Test"
        )
        
        high_violations = assessment.get_violations_by_severity(ViolationSeverity.HIGH)
        medium_violations = assessment.get_violations_by_severity(ViolationSeverity.MEDIUM)
        low_violations = assessment.get_violations_by_severity(ViolationSeverity.LOW)
        
        assert len(high_violations) == 1
        assert len(medium_violations) == 1
        assert len(low_violations) == 0  # Not violated
    
    def test_get_highest_severity(self):
        """Test getting highest severity violation."""
        violations = [
            ViolationResult("test1", True, 0.8, "Medium violation", ViolationSeverity.MEDIUM, ActionType.WARN),
            ViolationResult("test2", True, 0.9, "High violation", ViolationSeverity.HIGH, ActionType.MODIFY),
            ViolationResult("test3", False, 0.7, "No violation", ViolationSeverity.CRITICAL, ActionType.REJECT)
        ]
        
        assessment = ConstitutionalAssessment(
            content="Test",
            violations=violations,
            overall_score=0.3,
            is_compliant=False,
            recommended_action=ActionType.MODIFY,
            final_content="Test"
        )
        
        highest_severity = assessment.get_highest_severity()
        assert highest_severity == ViolationSeverity.HIGH


class TestConstitutionalPrincipleLibrary:
    """Test the ConstitutionalPrincipleLibrary class."""
    
    def test_get_default_principles(self):
        """Test getting default principles."""
        principles = ConstitutionalPrincipleLibrary.get_default_principles()
        
        assert len(principles) > 0
        assert all(isinstance(p, ConstitutionalPrinciple) for p in principles)
        
        # Check that we have principles of different types
        principle_types = {p.principle_type for p in principles}
        assert PrincipleType.SAFETY in principle_types
        assert PrincipleType.ETHICS in principle_types
        assert PrincipleType.PRIVACY in principle_types
    
    def test_get_principles_by_type(self):
        """Test filtering principles by type."""
        safety_principles = ConstitutionalPrincipleLibrary.get_principles_by_type(PrincipleType.SAFETY)
        ethics_principles = ConstitutionalPrincipleLibrary.get_principles_by_type(PrincipleType.ETHICS)
        
        assert all(p.principle_type == PrincipleType.SAFETY for p in safety_principles)
        assert all(p.principle_type == PrincipleType.ETHICS for p in ethics_principles)
    
    def test_get_principles_by_severity(self):
        """Test filtering principles by severity."""
        critical_principles = ConstitutionalPrincipleLibrary.get_principles_by_severity(ViolationSeverity.CRITICAL)
        high_principles = ConstitutionalPrincipleLibrary.get_principles_by_severity(ViolationSeverity.HIGH)
        
        assert all(p.severity == ViolationSeverity.CRITICAL for p in critical_principles)
        assert all(p.severity == ViolationSeverity.HIGH for p in high_principles)
    
    def test_save_and_load_principles(self):
        """Test saving and loading principles to/from file."""
        principles = ConstitutionalPrincipleLibrary.get_default_principles()[:3]  # Use first 3
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save principles
            ConstitutionalPrincipleLibrary.save_principles_to_file(principles, temp_file)
            
            # Load principles
            loaded_principles = ConstitutionalPrincipleLibrary.load_principles_from_file(temp_file)
            
            assert len(loaded_principles) == len(principles)
            
            for original, loaded in zip(principles, loaded_principles):
                assert original.id == loaded.id
                assert original.name == loaded.name
                assert original.principle_type == loaded.principle_type
                assert original.severity == loaded.severity
                
        finally:
            os.unlink(temp_file)


class TestLLMConstitutionalValidator:
    """Test the LLMConstitutionalValidator class."""
    
    @patch('constitutional_agent.ChatOpenAI')
    def test_validator_success(self, mock_chat_openai):
        """Test successful validation."""
        # Create a mock structured response
        from pydantic import BaseModel
        class MockValidationResponse(BaseModel):
            violated: bool
            confidence: float
            explanation: str
        
        mock_response = MockValidationResponse(
            violated=True,
            confidence=0.85,
            explanation="Contains harmful content"
        )
        
        # Mock the structured output
        mock_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = mock_response
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_chat_openai.return_value = mock_model
        
        validator = LLMConstitutionalValidator(mock_model)
        
        principle = ConstitutionalPrinciple(
            id="test",
            name="Test",
            description="Test principle",
            principle_type=PrincipleType.SAFETY,
            severity=ViolationSeverity.HIGH,
            action=ActionType.MODIFY,
            prompt_template="Test: {content}"
        )
        
        result = validator.validate("harmful content", principle)
        
        assert result.principle_id == "test"
        assert result.violated is True
        assert result.confidence == 0.85
        assert "harmful content" in result.explanation
    
    @patch('constitutional_agent.ChatOpenAI')
    def test_validator_json_parse_error(self, mock_chat_openai):
        """Test validator with structured output error."""
        # Mock the structured output to raise an exception
        mock_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.side_effect = Exception("Structured output failed")
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_chat_openai.return_value = mock_model
        
        validator = LLMConstitutionalValidator(mock_model)
        
        principle = ConstitutionalPrinciple(
            id="test",
            name="Test",
            description="Test principle",
            principle_type=PrincipleType.SAFETY,
            severity=ViolationSeverity.HIGH,
            action=ActionType.MODIFY,
            prompt_template="Test: {content}"
        )
        
        result = validator.validate("test content", principle)
        
        assert result.principle_id == "test"
        assert result.violated is False  # Fallback to False on error
        assert result.confidence == 0.0  # Fallback confidence
        assert "Validation failed due to error" in result.explanation


class TestLLMConstitutionalModifier:
    """Test the LLMConstitutionalModifier class."""
    
    @patch('constitutional_agent.ChatOpenAI')
    def test_modifier_success(self, mock_chat_openai):
        """Test successful content modification."""
        # Mock the model response
        mock_response = MagicMock()
        mock_response.content = "Modified safe content"
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model
        
        modifier = LLMConstitutionalModifier(mock_model)
        
        violation = ViolationResult(
            principle_id="safety_test",
            violated=True,
            confidence=0.9,
            explanation="Contains harmful instructions",
            severity=ViolationSeverity.HIGH,
            suggested_action=ActionType.MODIFY
        )
        
        result = modifier.modify("harmful content", violation)
        
        assert result == "Modified safe content"
    
    @patch('constitutional_agent.ChatOpenAI')
    def test_modifier_error(self, mock_chat_openai):
        """Test modifier with error."""
        # Mock the model to raise an exception
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_model
        
        modifier = LLMConstitutionalModifier(mock_model)
        
        violation = ViolationResult(
            principle_id="test",
            violated=True,
            confidence=0.8,
            explanation="Test violation",
            severity=ViolationSeverity.MEDIUM,
            suggested_action=ActionType.MODIFY
        )
        
        result = modifier.modify("test content", violation)
        
        assert "constitutional guidelines" in result
        assert "API Error" in result


class TestConstitutionalAI:
    """Test the ConstitutionalAI class."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_agent_initialization(self):
        """Test constitutional AI agent initialization."""
        agent = ConstitutionalAI()
        
        assert agent.model is not None
        assert len(agent.principles) > 0
        assert agent.validator is not None
        assert agent.modifier is not None
        assert agent.graph is not None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_add_remove_principle(self):
        """Test adding and removing principles."""
        agent = ConstitutionalAI()
        
        initial_count = len(agent.principles)
        
        # Add principle
        new_principle = ConstitutionalPrinciple(
            id="test_new",
            name="Test New",
            description="Test principle",
            principle_type=PrincipleType.SAFETY,
            severity=ViolationSeverity.MEDIUM,
            action=ActionType.WARN,
            prompt_template="Test: {content}"
        )
        
        agent.add_principle(new_principle)
        assert len(agent.principles) == initial_count + 1
        
        # Remove principle
        agent.remove_principle("test_new")
        assert len(agent.principles) == initial_count
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_enable_disable_principle(self):
        """Test enabling and disabling principles."""
        agent = ConstitutionalAI()
        
        # Get first principle
        principle_id = agent.principles[0].id
        
        # Disable principle
        agent.disable_principle(principle_id)
        principle = next(p for p in agent.principles if p.id == principle_id)
        assert principle.enabled is False
        
        # Enable principle
        agent.enable_principle(principle_id)
        principle = next(p for p in agent.principles if p.id == principle_id)
        assert principle.enabled is True
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_get_principles_summary(self):
        """Test getting principles summary."""
        agent = ConstitutionalAI()
        
        summary = agent.get_principles_summary()
        
        assert "total_principles" in summary
        assert "enabled_principles" in summary
        assert "disabled_principles" in summary
        assert "principles_by_type" in summary
        
        assert summary["total_principles"] > 0
        assert isinstance(summary["principles_by_type"], dict)


class TestConstitutionalResponse:
    """Test the ConstitutionalResponse class."""
    
    def test_response_creation(self):
        """Test creating a constitutional response."""
        assessment = ConstitutionalAssessment(
            content="Test content",
            violations=[],
            overall_score=0.9,
            is_compliant=True,
            recommended_action=ActionType.LOG,
            final_content="Test content"
        )
        
        response = ConstitutionalResponse(
            original_content="Original content",
            final_content="Final content",
            assessment=assessment,
            was_modified=True,
            violations_found=0,
            compliance_score=0.9
        )
        
        assert response.original_content == "Original content"
        assert response.final_content == "Final content"
        assert response.assessment == assessment
        assert response.was_modified is True
        assert response.violations_found == 0
        assert response.compliance_score == 0.9


if __name__ == "__main__":
    pytest.main([__file__])
