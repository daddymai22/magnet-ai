import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Strategy:
    """Represents a learned strategy"""
    strategy_id: str
    description: str
    effectiveness: float
    episodes_discovered: int
    timestamp: Optional[str]
    agent_id: Optional[int] = None
    confidence: float = 0.5
    usage_count: int = 0

@dataclass
class AgentInsight:
    """Individual agent's learning insight"""
    agent_id: int
    agent_name: str
    insight_type: str  # 'strategy', 'pattern', 'anomaly'
    content: str
    effectiveness: float
    timestamp: str
    episode_discovered: int

class TrainingDataManager:
    """Manages training data, insights, and shared memory"""
    
    def __init__(self, data_file: str = "training_data.json"):
        self.data_file = Path(data_file)
        self.data = self._load_data()
        self.instance_id = self._generate_instance_id()
        logger.info(f"🧠 Training Data Manager initialized (Instance: {self.instance_id})")
    
    def _generate_instance_id(self) -> str:
        """Generate unique instance ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _load_data(self) -> Dict:
        """Load training data from file"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")
        return self._default_data()
    
    def _default_data(self) -> Dict:
        """Return default training data structure"""
        return {
            "training_metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "training_steps_total": 0,
                "total_episodes": 0,
                "model_instances": 1
            },
            "learning_insights": {
                "hider_strategies": [],
                "seeker_strategies": []
            },
            "agent_performance": {},
            "shared_memory": {
                "global_insights": [],
                "best_hider_techniques": [],
                "best_seeker_techniques": [],
                "anomaly_patterns": [],
                "reward_hacking_attempts": [],
                "last_sync": None,
                "sync_frequency_seconds": 300,
                "instances_connected": 1
            },
            "training_episodes": []
        }
    
    def save_data(self) -> None:
        """Save training data to file"""
        self.data["training_metadata"]["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"✅ Training data saved")
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
    
    def record_episode(self, agent_id: int, agent_name: str, episode_data: Dict) -> None:
        """Record a training episode"""
        episode = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "agent_name": agent_name,
            "episode_number": self.data["training_metadata"]["total_episodes"],
            "data": episode_data
        }
        
        self.data["training_episodes"].append(episode)
        self.data["training_metadata"]["total_episodes"] += 1
        
        logger.info(f"📊 Episode {episode['episode_number']} recorded for {agent_name}")
    
    def add_strategy(self, agent_id: int, agent_name: str, strategy: Strategy) -> None:
        """Add a learned strategy to training data"""
        strategy_dict = asdict(strategy)
        strategy_dict["agent_id"] = agent_id
        strategy_dict["agent_name"] = agent_name
        strategy_dict["timestamp"] = datetime.now().isoformat()
        
        # Determine strategy type
        strategy_type = "hider_strategies" if "hider" in agent_name.lower() else "seeker_strategies"
        
        self.data["learning_insights"][strategy_type].append(strategy_dict)
        logger.info(f"🎯 Strategy '{strategy.strategy_id}' learned by {agent_name} (effectiveness: {strategy.effectiveness:.2f})")
    
    def update_agent_performance(self, agent_id: int, agent_name: str, metrics: Dict) -> None:
        """Update agent performance metrics"""
        if agent_name not in self.data["agent_performance"]:
            self.data["agent_performance"][agent_name] = {
                "total_episodes": 0,
                "average_survival_steps": 0.0,
                "win_rate": 0.0,
                "learned_strategies": [],
                "last_update": None,
                "contribution_to_shared_memory": 0
            }
        
        self.data["agent_performance"][agent_name].update(metrics)
        self.data["agent_performance"][agent_name]["last_update"] = datetime.now().isoformat()
        
        logger.info(f"📈 Performance updated for {agent_name}: {metrics}")
    
    def add_global_insight(self, insight: AgentInsight) -> None:
        """Add insight to shared global memory"""
        insight_dict = asdict(insight)
        self.data["shared_memory"]["global_insights"].append(insight_dict)
        logger.info(f"💡 Global insight added: {insight.content[:50]}...")
    
    def add_best_technique(self, agent_name: str, technique: Dict, technique_type: str = "hider") -> None:
        """Add best technique to shared memory"""
        technique["agent_name"] = agent_name
        technique["timestamp"] = datetime.now().isoformat()
        
        key = "best_hider_techniques" if technique_type == "hider" else "best_seeker_techniques"
        self.data["shared_memory"][key].append(technique)
        logger.info(f"⭐ Best technique from {agent_name} added to shared memory")
    
    def add_anomaly_pattern(self, description: str, frequency: int, agents_detected: List[int]) -> None:
        """Record anomaly pattern detected by snitch"""
        pattern = {
            "description": description,
            "frequency": frequency,
            "agents_detected": agents_detected,
            "timestamp": datetime.now().isoformat()
        }
        self.data["shared_memory"]["anomaly_patterns"].append(pattern)
        logger.info(f"🚨 Anomaly pattern recorded: {description}")
    
    def get_shared_memory(self) -> Dict:
        """Get current shared memory state"""
        return self.data["shared_memory"]
    
    def get_agent_performance(self, agent_name: Optional[str] = None) -> Dict:
        """Get agent performance data"""
        if agent_name:
            return self.data["agent_performance"].get(agent_name, {})
        return self.data["agent_performance"]
    
    def sync_shared_memory(self) -> Dict:
        """Sync and return shared memory for inter-AI communication"""
        self.data["shared_memory"]["last_sync"] = datetime.now().isoformat()
        self.save_data()
        return self.get_shared_memory()
    
    def apply_shared_insights(self, shared_memory: Dict) -> None:
        """Apply shared memory insights to improve local agent"""
        logger.info("🔄 Applying shared insights...")
        
        # Add global insights
        for insight in shared_memory.get("global_insights", []):
            if insight not in self.data["shared_memory"]["global_insights"]:
                self.data["shared_memory"]["global_insights"].append(insight)
        
        # Add best techniques
        for technique in shared_memory.get("best_hider_techniques", []):
            if technique not in self.data["shared_memory"]["best_hider_techniques"]:
                self.data["shared_memory"]["best_hider_techniques"].append(technique)
        
        for technique in shared_memory.get("best_seeker_techniques", []):
            if technique not in self.data["shared_memory"]["best_seeker_techniques"]:
                self.data["shared_memory"]["best_seeker_techniques"].append(technique)
        
        # Add anomaly patterns
        for pattern in shared_memory.get("anomaly_patterns", []):
            if pattern not in self.data["shared_memory"]["anomaly_patterns"]:
                self.data["shared_memory"]["anomaly_patterns"].append(pattern)
        
        logger.info(f"✅ Synced {len(shared_memory.get('global_insights', []))} global insights")
        self.save_data()
    
    def get_training_summary(self) -> Dict:
        """Get summary of training progress"""
        return {
            "version": self.data["training_metadata"]["version"],
            "total_training_steps": self.data["training_metadata"]["training_steps_total"],
            "total_episodes": self.data["training_metadata"]["total_episodes"],
            "last_updated": self.data["training_metadata"]["last_updated"],
            "agents": list(self.data["agent_performance"].keys()),
            "global_insights_count": len(self.data["shared_memory"]["global_insights"]),
            "anomalies_detected": len(self.data["shared_memory"]["anomaly_patterns"])
        }

class KnowledgeSharer:
    """Shares knowledge between AI instances"""
    
    def __init__(self):
        self.training_manager = TrainingDataManager()
        logger.info("📡 Knowledge Sharer initialized")
    
    def extract_learnings(self, episode_data: Dict, agent_name: str) -> Dict:
        """Extract key learnings from episode"""
        learnings = {
            "strategies": [],
            "patterns": [],
            "effectiveness_metrics": {}
        }
        
        # Extract strategy effectiveness
        if "reward" in episode_data:
            reward = episode_data["reward"]
            if reward > 0.5:
                learnings["effectiveness_metrics"]["high_reward_episode"] = True
                learnings["strategies"].append({
                    "type": "winning_strategy",
                    "reward": reward,
                    "steps": episode_data.get("steps", 0)
                })
        
        # Extract patterns
        if "positions" in episode_data:
            learnings["patterns"].append({
                "type": "movement_pattern",
                "data": episode_data["positions"]
            })
        
        return learnings
    
    def broadcast_learning(self, agent_id: int, agent_name: str, learning_data: Dict) -> None:
        """Broadcast learning to shared memory"""
        insight = AgentInsight(
            agent_id=agent_id,
            agent_name=agent_name,
            insight_type="strategy",
            content=f"Learned from episode: {learning_data}",
            effectiveness=learning_data.get("effectiveness_metrics", {}).get("reward", 0.0),
            timestamp=datetime.now().isoformat(),
            episode_discovered=self.training_manager.data["training_metadata"]["total_episodes"]
        )
        self.training_manager.add_global_insight(insight)
    
    def receive_broadcast(self) -> Dict:
        """Receive broadcast from other AI instances"""
        return self.training_manager.get_shared_memory()

# Example usage
if __name__ == "__main__":
    manager = TrainingDataManager()
    
    # Simulate learning
    logger.info("\nSimulating AI Learning Process...\n")
    
    # Agent 1 learns a strategy
    strategy = Strategy(
        strategy_id="test_strategy_1",
        description="Test hiding strategy",
        effectiveness=0.75,
        episodes_discovered=1,
        timestamp=datetime.now().isoformat()
    )
    manager.add_strategy(0, "student_hider_1", strategy)
    
    # Add to shared memory
    insight = AgentInsight(
        agent_id=0,
        agent_name="student_hider_1",
        insight_type="strategy",
        content="Corner hiding reduces visibility by 40%",
        effectiveness=0.75,
        timestamp=datetime.now().isoformat(),
        episode_discovered=0
    )
    manager.add_global_insight(insight)
    
    # Update performance
    manager.update_agent_performance(0, "student_hider_1", {
        "total_episodes": 10,
        "average_survival_steps": 150.5,
        "win_rate": 0.6,
        "contribution_to_shared_memory": 3
    })
    
    manager.save_data()
    
    # Get summary
    summary = manager.get_training_summary()
    logger.info(f"\nTraining Summary:\n{json.dumps(summary, indent=2)}")
