import numpy as np
import gym
from gym import spaces
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math

class AgentType(Enum):
    HIDER = 0
    SEEKER = 1
    SNITCH = 2

@dataclass
class Agent:
    """Represents a single agent in the hide-and-seek environment"""
    agent_id: int
    agent_type: AgentType
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    found: bool = False
    observation_history: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.observation_history is None:
            self.observation_history = []
    
    def distance_to(self, other_x: float, other_y: float) -> float:
        """Calculate Euclidean distance to a point"""
        return math.sqrt((self.x - other_x)**2 + (self.y - other_y)**2)
    
    def update_position(self, action: np.ndarray, dt: float = 0.05, max_speed: float = 5.0):
        """Update agent position based on action and physics"""
        # Action: [forward/backward, left/right, jump]
        acceleration_x = action[0] * 10.0
        acceleration_y = action[1] * 10.0
        
        # Update velocity with friction
        self.vx = self.vx * 0.95 + acceleration_x * dt
        self.vy = self.vy * 0.95 + acceleration_y * dt
        
        # Clamp speed
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

class HideAndSeekEnv(gym.Env):
    """Multi-agent hide-and-seek environment for Magnet AI training"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.map_size = config['environment']['map_size']
        self.episode_length = config['environment']['episode_length']
        self.num_hiders = config['environment']['num_hiders']
        self.num_seekers = config['environment']['num_seekers']
        self.total_agents = self.num_hiders + self.num_seekers + 1  # +1 for snitch
        
        # Initialize agents
        self.agents: Dict[int, Agent] = {}
        self.snitch_agent: Agent = None
        self._init_agents()
        
        # Game state
        self.step_count = 0
        self.episode_rewards = {i: 0.0 for i in range(self.total_agents)}
        self.episode_data = {i: [] for i in range(self.total_agents)}
        
        # Action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # [x, y, jump]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._get_obs_size(),), dtype=np.float32)
    
    def _init_agents(self):
        """Initialize all agents in the environment"""
        agent_id = 0
        
        # Create hiders
        for i in range(self.num_hiders):
            self.agents[agent_id] = Agent(
                agent_id=agent_id,
                agent_type=AgentType.HIDER,
                x=np.random.uniform(-self.map_size/2, self.map_size/2),
                y=np.random.uniform(-self.map_size/2, self.map_size/2)
            )
            agent_id += 1
        
        # Create seekers
        for i in range(self.num_seekers):
            self.agents[agent_id] = Agent(
                agent_id=agent_id,
                agent_type=AgentType.SEEKER,
                x=0,  # Spawn seekers in center
                y=0
            )
            agent_id += 1
        
        # Create snitch agent (monitors for rule violations)
        self.snitch_agent = Agent(
            agent_id=agent_id,
            agent_type=AgentType.SNITCH,
            x=np.random.uniform(-self.map_size/2, self.map_size/2),
            y=np.random.uniform(-self.map_size/2, self.map_size/2)
        )
    
    def _get_obs_size(self) -> int:
        """Calculate observation vector size"""
        # Own position (2) + velocity (2) + found status (1)
        # + All other agents positions (2 * num_agents) + distances (num_agents)
        return 5 + (2 * (self.total_agents - 1)) + (self.total_agents - 1)
    
    def _get_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent"""
        agent = self.agents[agent_id]
        obs = []
        
        # Own state
        obs.extend([agent.x / self.map_size, agent.y / self.map_size])  # Normalized position
        obs.extend([agent.vx / 5.0, agent.vy / 5.0])  # Normalized velocity
        obs.append(1.0 if agent.found else 0.0)  # Found status
        
        # Relative observations of other agents
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id:
                rel_x = (other_agent.x - agent.x) / self.map_size
                rel_y = (other_agent.y - agent.y) / self.map_size
                obs.extend([rel_x, rel_y])
                
                distance = agent.distance_to(other_agent.x, other_agent.y) / self.map_size
                obs.append(distance)
        
        # Add snitch observations
        rel_x = (self.snitch_agent.x - agent.x) / self.map_size
        rel_y = (self.snitch_agent.y - agent.y) / self.map_size
        obs.extend([rel_x, rel_y])
        distance = agent.distance_to(self.snitch_agent.x, self.snitch_agent.y) / self.map_size
        obs.append(distance)
        
        return np.array(obs, dtype=np.float32)
    
    def _check_collisions(self) -> Dict[int, bool]:
        """Check for collisions between seekers and hiders"""
        caught = {}
        touch_distance = 2.0  # Collision radius
        
        for seeker_id, seeker in self.agents.items():
            if seeker.agent_type != AgentType.SEEKER:
                continue
            
            for hider_id, hider in self.agents.items():
                if hider.agent_type != AgentType.HIDER:
                    continue
                
                distance = seeker.distance_to(hider.x, hider.y)
                if distance < touch_distance and not hider.found:
                    hider.found = True
                    caught[hider_id] = True
        
        return caught
    
    def _detect_reward_hacking(self) -> Dict[int, float]:
        """Detect potential reward hacking behaviors (Snitch AI)"""
        suspicious_scores = {}
        
        # Check for wall-clipping (moving outside bounds too quickly)
        for agent_id, agent in self.agents.items():
            if abs(agent.x) > self.map_size/2 or abs(agent.y) > self.map_size/2:
                agent.x = np.clip(agent.x, -self.map_size/2, self.map_size/2)
                agent.y = np.clip(agent.y, -self.map_size/2, self.map_size/2)
                suspicious_scores[agent_id] = 0.8  # High suspicion
        
        return suspicious_scores
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1
        rewards = {i: 0.0 for i in range(self.total_agents)}
        dones = {i: False for i in range(self.total_agents)}
        info = {i: {} for i in range(self.total_agents)}
        
        # Update agent positions
        for agent_id, action in actions.items():
            if agent_id in self.agents and not self.agents[agent_id].found:
                self.agents[agent_id].update_position(action)
        
        # Check collisions
        caught = self._check_collisions()
        
        # Reward hiders for staying unfound
        for hider_id, hider in self.agents.items():
            if hider.agent_type == AgentType.HIDER:
                if hider.found:
                    rewards[hider_id] = -1.0
                    info[hider_id]['caught'] = True
                else:
                    rewards[hider_id] = 0.1  # Reward for surviving each step
        
        # Reward seekers for finding hiders
        for seeker_id, seeker in self.agents.items():
            if seeker.agent_type == AgentType.SEEKER:
                num_found = sum(1 for h in self.agents.values() if h.agent_type == AgentType.HIDER and h.found)
                rewards[seeker_id] = num_found * 0.5
        
        # Snitch detects anomalies
        suspicious = self._detect_reward_hacking()
        for agent_id, score in suspicious.items():
            if score > self.config['safety']['snitch_threshold']:
                info[agent_id]['suspicious'] = True
                # Penalty for suspicious behavior
                rewards[agent_id] -= 0.5
        
        # Episode termination
        all_hiders_found = all(h.found for h in self.agents.values() if h.agent_type == AgentType.HIDER)
        if self.step_count >= self.episode_length or all_hiders_found:
            for agent_id in self.agents.keys():
                dones[agent_id] = True
        
        # Get observations
        observations = {agent_id: self._get_observation(agent_id) for agent_id in self.agents.keys()}
        
        # Update rewards tracking
        for agent_id, reward in rewards.items():
            self.episode_rewards[agent_id] += reward
            self.episode_data[agent_id].append({
                'step': self.step_count,
                'reward': reward,
                'position': (self.agents[agent_id].x, self.agents[agent_id].y),
                'suspicious': info[agent_id].get('suspicious', False)
            })
        
        return observations, rewards, dones, info
    
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset the environment for a new episode"""
        self.step_count = 0
        self.episode_rewards = {i: 0.0 for i in range(self.total_agents)}
        self.episode_data = {i: [] for i in range(self.total_agents)}
        
        # Reset agents
        for agent in self.agents.values():
            agent.found = False
            agent.x = np.random.uniform(-self.map_size/2, self.map_size/2)
            agent.y = np.random.uniform(-self.map_size/2, self.map_size/2)
            agent.vx = 0.0
            agent.vy = 0.0
        
        # Seekers spawn in center
        for agent in self.agents.values():
            if agent.agent_type == AgentType.SEEKER:
                agent.x = 0
                agent.y = 0
        
        observations = {agent_id: self._get_observation(agent_id) for agent_id in self.agents.keys()}
        return observations
    
    def render(self, mode='human'):
        """Render the environment (optional visualization)"""
        pass
    
    def close(self):
        """Clean up resources"""
        pass
