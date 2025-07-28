"""Constitutional AI models and principles."""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from datetime import datetime
from pydantic import BaseModel, Field


class PrincipleType(Enum):
    """Types of constitutional principles."""
    SAFETY = "safety"
    ETHICS = "ethics"
    PRIVACY = "privacy"
    ACCURACY = "accuracy"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"


class ViolationSeverity(Enum):
    """Severity levels for principle violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of actions to take when violations are detected."""
    WARN = "warn"
    MODIFY = "modify"
    REJECT = "reject"
    ESCALATE = "escalate"
    LOG = "log"


@dataclass
class ConstitutionalPrinciple:
    """A constitutional principle that guides AI behavior."""
    id: str
    name: str
    description: str
    principle_type: PrincipleType
    severity: ViolationSeverity
    action: ActionType
    prompt_template: str
    examples: List[Dict[str, str]] = field(default_factory=list)
    enabled: bool = True
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert principle to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "principle_type": self.principle_type.value,
            "severity": self.severity.value,
            "action": self.action.value,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "enabled": self.enabled,
            "weight": self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConstitutionalPrinciple':
        """Create principle from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            principle_type=PrincipleType(data["principle_type"]),
            severity=ViolationSeverity(data["severity"]),
            action=ActionType(data["action"]),
            prompt_template=data["prompt_template"],
            examples=data.get("examples", []),
            enabled=data.get("enabled", True),
            weight=data.get("weight", 1.0)
        )


@dataclass
class ViolationResult:
    """Result of a constitutional principle evaluation."""
    principle_id: str
    violated: bool
    confidence: float
    explanation: str
    severity: ViolationSeverity
    suggested_action: ActionType
    modified_content: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "principle_id": self.principle_id,
            "violated": self.violated,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "severity": self.severity.value,
            "suggested_action": self.suggested_action.value,
            "modified_content": self.modified_content
        }


@dataclass
class ConstitutionalAssessment:
    """Complete constitutional assessment of content."""
    content: str
    violations: List[ViolationResult]
    overall_score: float
    is_compliant: bool
    recommended_action: ActionType
    final_content: str
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    
    def get_violations_by_severity(self, severity: ViolationSeverity) -> List[ViolationResult]:
        """Get violations by severity level."""
        return [v for v in self.violations if v.violated and v.severity == severity]
    
    def get_highest_severity(self) -> Optional[ViolationSeverity]:
        """Get the highest severity violation."""
        if not self.violations:
            return None
        
        violated = [v for v in self.violations if v.violated]
        if not violated:
            return None
        
        severity_order = [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH, 
                         ViolationSeverity.MEDIUM, ViolationSeverity.LOW]
        
        for severity in severity_order:
            if any(v.severity == severity for v in violated):
                return severity
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "content": self.content,
            "violations": [v.to_dict() for v in self.violations],
            "overall_score": self.overall_score,
            "is_compliant": self.is_compliant,
            "recommended_action": self.recommended_action.value,
            "final_content": self.final_content,
            "assessment_timestamp": self.assessment_timestamp.isoformat()
        }


class ConstitutionalPrincipleLibrary:
    """Library of predefined constitutional principles."""
    
    @staticmethod
    def get_default_principles() -> List[ConstitutionalPrinciple]:
        """Get a set of default constitutional principles."""
        return [
            # Safety principles
            ConstitutionalPrinciple(
                id="safety_harmful_content",
                name="Harmful Content Prevention",
                description="Prevent generation of content that could cause physical or psychological harm",
                principle_type=PrincipleType.SAFETY,
                severity=ViolationSeverity.CRITICAL,
                action=ActionType.REJECT,
                prompt_template="Does this content contain instructions or encouragement for harmful activities, violence, self-harm, or dangerous behaviors? Content: {content}",
                examples=[
                    {"violation": "How to make explosives", "explanation": "Contains dangerous instructions"},
                    {"compliant": "How to safely handle chemicals in a lab", "explanation": "Promotes safety"}
                ]
            ),
            
            # Privacy principles
            ConstitutionalPrinciple(
                id="privacy_personal_info",
                name="Personal Information Protection",
                description="Protect personal and sensitive information from being disclosed",
                principle_type=PrincipleType.PRIVACY,
                severity=ViolationSeverity.HIGH,
                action=ActionType.MODIFY,
                prompt_template="Does this content contain personal information like names, addresses, phone numbers, email addresses, or other identifying information? Content: {content}",
                examples=[
                    {"violation": "John Smith lives at 123 Main St", "explanation": "Contains personal address"},
                    {"compliant": "A person lives in the downtown area", "explanation": "General location only"}
                ]
            ),
            
            # Ethics principles
            ConstitutionalPrinciple(
                id="ethics_discrimination",
                name="Anti-Discrimination",
                description="Prevent discriminatory content based on race, gender, religion, or other protected characteristics",
                principle_type=PrincipleType.ETHICS,
                severity=ViolationSeverity.HIGH,
                action=ActionType.MODIFY,
                prompt_template="Does this content contain discriminatory language or promote bias against any group based on race, gender, religion, nationality, or other characteristics? Content: {content}",
                examples=[
                    {"violation": "People from X country are lazy", "explanation": "Contains ethnic stereotyping"},
                    {"compliant": "Different cultures have different work styles", "explanation": "Acknowledges diversity respectfully"}
                ]
            ),
            
            # Accuracy principles
            ConstitutionalPrinciple(
                id="accuracy_misinformation",
                name="Misinformation Prevention",
                description="Prevent spread of false or misleading information",
                principle_type=PrincipleType.ACCURACY,
                severity=ViolationSeverity.MEDIUM,
                action=ActionType.WARN,
                prompt_template="Does this content contain factual claims that are demonstrably false or misleading? Content: {content}",
                examples=[
                    {"violation": "The Earth is flat", "explanation": "Scientifically false claim"},
                    {"compliant": "Some people believe the Earth is flat, but scientific evidence shows it's spherical", "explanation": "Acknowledges belief while stating facts"}
                ]
            ),
            
            # Fairness principles
            ConstitutionalPrinciple(
                id="fairness_balanced_perspective",
                name="Balanced Perspective",
                description="Provide balanced and fair representation of different viewpoints",
                principle_type=PrincipleType.FAIRNESS,
                severity=ViolationSeverity.MEDIUM,
                action=ActionType.MODIFY,
                prompt_template="Does this content present a one-sided view on a controversial topic without acknowledging other perspectives? Content: {content}",
                examples=[
                    {"violation": "Policy X is completely wrong and harmful", "explanation": "One-sided political statement"},
                    {"compliant": "Policy X has both supporters and critics, with different perspectives on its impact", "explanation": "Acknowledges multiple viewpoints"}
                ]
            ),
            
            # Transparency principles
            ConstitutionalPrinciple(
                id="transparency_ai_disclosure",
                name="AI Disclosure",
                description="Be transparent about AI capabilities and limitations",
                principle_type=PrincipleType.TRANSPARENCY,
                severity=ViolationSeverity.LOW,
                action=ActionType.MODIFY,
                prompt_template="Does this content claim capabilities or knowledge that an AI system shouldn't claim to have? Content: {content}",
                examples=[
                    {"violation": "I personally experienced this event", "explanation": "AI claiming personal experience"},
                    {"compliant": "Based on available information, this event occurred", "explanation": "Appropriate AI response"}
                ]
            ),
            
            # Helpfulness principles
            ConstitutionalPrinciple(
                id="helpfulness_constructive",
                name="Constructive Assistance",
                description="Provide helpful and constructive responses",
                principle_type=PrincipleType.HELPFULNESS,
                severity=ViolationSeverity.LOW,
                action=ActionType.MODIFY,
                prompt_template="Is this content helpful and constructive for the user's request? Content: {content}",
                examples=[
                    {"violation": "I can't help with that", "explanation": "Unhelpful dismissal"},
                    {"compliant": "I can't help with that specific request, but here's what I can do instead", "explanation": "Offers alternatives"}
                ]
            ),
            
            # Harmlessness principles
            ConstitutionalPrinciple(
                id="harmlessness_emotional_harm",
                name="Emotional Harm Prevention",
                description="Avoid content that could cause emotional distress or psychological harm",
                principle_type=PrincipleType.HARMLESSNESS,
                severity=ViolationSeverity.MEDIUM,
                action=ActionType.MODIFY,
                prompt_template="Could this content cause emotional distress, anxiety, or psychological harm to vulnerable individuals? Content: {content}",
                examples=[
                    {"violation": "You're worthless and will never succeed", "explanation": "Emotionally harmful language"},
                    {"compliant": "Everyone faces challenges, and with effort, improvement is possible", "explanation": "Supportive and encouraging"}
                ]
            )
        ]
    
    @staticmethod
    def get_principles_by_type(principle_type: PrincipleType) -> List[ConstitutionalPrinciple]:
        """Get principles filtered by type."""
        all_principles = ConstitutionalPrincipleLibrary.get_default_principles()
        return [p for p in all_principles if p.principle_type == principle_type]
    
    @staticmethod
    def get_principles_by_severity(severity: ViolationSeverity) -> List[ConstitutionalPrinciple]:
        """Get principles filtered by severity."""
        all_principles = ConstitutionalPrincipleLibrary.get_default_principles()
        return [p for p in all_principles if p.severity == severity]
    
    @staticmethod
    def save_principles_to_file(principles: List[ConstitutionalPrinciple], filename: str):
        """Save principles to JSON file."""
        data = [p.to_dict() for p in principles]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_principles_from_file(filename: str) -> List[ConstitutionalPrinciple]:
        """Load principles from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        return [ConstitutionalPrinciple.from_dict(item) for item in data]


class ConstitutionalValidator(ABC):
    """Abstract base class for constitutional validators."""
    
    @abstractmethod
    def validate(self, content: str, principle: ConstitutionalPrinciple) -> ViolationResult:
        """Validate content against a constitutional principle."""
        pass


# Pydantic models for structured outputs
class ConstitutionalViolationModel(BaseModel):
    """Pydantic model for constitutional violation results."""
    principle_id: str
    violated: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    severity: str = Field(..., pattern=r"^(low|medium|high|critical)$")
    suggested_action: str = Field(..., pattern=r"^(warn|modify|reject|escalate|log)$")
    modified_content: Optional[str] = None


class ConstitutionalEvaluationResponse(BaseModel):
    """Structured response for constitutional evaluation."""
    violations: List[ConstitutionalViolationModel]
    overall_compliant: bool
    recommended_action: str = Field(..., pattern=r"^(warn|modify|reject|escalate|log)$")
    final_content: str


class ConstitutionalModifier(ABC):
    """Abstract base class for constitutional content modifiers."""
    
    @abstractmethod
    def modify(self, content: str, violation: ViolationResult) -> str:
        """Modify content to address constitutional violation."""
        pass
