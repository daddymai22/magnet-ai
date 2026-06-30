from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import json

class FeedbackType(Enum):
    """Types of feedback for AI training"""
    POSITIVE = "positive"  # Like ✓
    NEGATIVE = "negative"  # Dislike ✗
    TEACHER_INPUT = "teacher"  # Teacher guidance
    SNITCH_ALERT = "snitch"  # Snitch detection

@dataclass
class Feedback:
    """Single feedback entry"""
    feedback_id: str
    agent_id: int
    feedback_type: FeedbackType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict = field(default_factory=dict)  # Episode data, action taken, etc.
    weight: float = 1.0  # How much this feedback impacts training
    
    def to_dict(self) -> dict:
        return {
            'feedback_id': self.feedback_id,
            'agent_id': self.agent_id,
            'feedback_type': self.feedback_type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'weight': self.weight
        }

@dataclass
class FeedbackSession:
    """Tracks all feedback for an agent session"""
    session_id: str
    agent_id: int
    start_time: datetime = field(default_factory=datetime.now)
    feedback_list: List[Feedback] = field(default_factory=list)
    total_positive: int = 0
    total_negative: int = 0
    teacher_inputs: List[str] = field(default_factory=list)
    snitch_alerts: int = 0
    
    def add_feedback(self, feedback: Feedback) -> None:
        """Add feedback to session"""
        self.feedback_list.append(feedback)
        
        if feedback.feedback_type == FeedbackType.POSITIVE:
            self.total_positive += 1
        elif feedback.feedback_type == FeedbackType.NEGATIVE:
            self.total_negative += 1
        elif feedback.feedback_type == FeedbackType.TEACHER_INPUT:
            self.teacher_inputs.append(feedback.content)
        elif feedback.feedback_type == FeedbackType.SNITCH_ALERT:
            self.snitch_alerts += 1
    
    def get_feedback_ratio(self) -> float:
        """Get positive/total feedback ratio (0.0 to 1.0)"""
        total = self.total_positive + self.total_negative
        if total == 0:
            return 0.5  # Neutral
        return self.total_positive / total
    
    def to_dict(self) -> dict:
        return {
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'start_time': self.start_time.isoformat(),
            'total_positive': self.total_positive,
            'total_negative': self.total_negative,
            'feedback_ratio': self.get_feedback_ratio(),
            'teacher_inputs': len(self.teacher_inputs),
            'snitch_alerts': self.snitch_alerts,
            'feedback_count': len(self.feedback_list)
        }

class FeedbackManager:
    """Manages all feedback across agents"""
    
    def __init__(self):
        self.sessions: Dict[str, FeedbackSession] = {}
        self.feedback_history: List[Feedback] = []
    
    def create_session(self, session_id: str, agent_id: int) -> FeedbackSession:
        """Create new feedback session"""
        session = FeedbackSession(session_id=session_id, agent_id=agent_id)
        self.sessions[session_id] = session
        return session
    
    def add_feedback(self, session_id: str, feedback: Feedback) -> bool:
        """Add feedback to a session"""
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].add_feedback(feedback)
        self.feedback_history.append(feedback)
        return True
    
    def like_response(self, session_id: str, agent_id: int, response_text: str) -> None:
        """User likes a response (Claude-style feedback)"""
        feedback = Feedback(
            feedback_id=f"like_{len(self.feedback_history)}",
            agent_id=agent_id,
            feedback_type=FeedbackType.POSITIVE,
            content=f"User liked response: {response_text[:100]}...",
            weight=1.0
        )
        self.add_feedback(session_id, feedback)
    
    def dislike_response(self, session_id: str, agent_id: int, response_text: str) -> None:
        """User dislikes a response (Claude-style feedback)"""
        feedback = Feedback(
            feedback_id=f"dislike_{len(self.feedback_history)}",
            agent_id=agent_id,
            feedback_type=FeedbackType.NEGATIVE,
            content=f"User disliked response: {response_text[:100]}...",
            weight=1.0
        )
        self.add_feedback(session_id, feedback)
    
    def teacher_input(self, session_id: str, agent_id: int, instruction: str) -> None:
        """Teacher provides training instruction (no character limit)"""
        feedback = Feedback(
            feedback_id=f"teacher_{len(self.feedback_history)}",
            agent_id=agent_id,
            feedback_type=FeedbackType.TEACHER_INPUT,
            content=instruction,
            weight=2.0  # Teacher feedback is weighted higher
        )
        self.add_feedback(session_id, feedback)
    
    def snitch_alert(self, session_id: str, agent_id: int, suspicious_behavior: str, confidence: float) -> None:
        """Snitch AI flags suspicious behavior"""
        feedback = Feedback(
            feedback_id=f"snitch_{len(self.feedback_history)}",
            agent_id=agent_id,
            feedback_type=FeedbackType.SNITCH_ALERT,
            content=f"Suspicious: {suspicious_behavior}",
            context={'confidence': confidence},
            weight=1.5  # Snitch alerts are important but lower than teacher
        )
        self.add_feedback(session_id, feedback)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get summary of feedback for a session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return session.to_dict()
    
    def export_feedback(self, output_path: str) -> None:
        """Export all feedback to JSON"""
        export_data = {
            'sessions': {sid: s.to_dict() for sid, s in self.sessions.items()},
            'total_feedback_entries': len(self.feedback_history),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def get_agent_stats(self, agent_id: int) -> Dict:
        """Get aggregated stats for an agent across all sessions"""
        agent_sessions = [s for s in self.sessions.values() if s.agent_id == agent_id]
        
        total_positive = sum(s.total_positive for s in agent_sessions)
        total_negative = sum(s.total_negative for s in agent_sessions)
        total_teacher_inputs = sum(len(s.teacher_inputs) for s in agent_sessions)
        total_snitch_alerts = sum(s.snitch_alerts for s in agent_sessions)
        
        return {
            'agent_id': agent_id,
            'sessions': len(agent_sessions),
            'total_positive_feedback': total_positive,
            'total_negative_feedback': total_negative,
            'feedback_ratio': total_positive / (total_positive + total_negative) if (total_positive + total_negative) > 0 else 0.5,
            'teacher_inputs_received': total_teacher_inputs,
            'snitch_alerts': total_snitch_alerts
        }
