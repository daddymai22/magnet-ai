import yaml
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from environment import HideAndSeekEnv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MagnetTrainingCallback(BaseCallback):
    """Custom callback for tracking training progress"""
    
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        Path(log_dir).mkdir(exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called after every environment step"""
        if self.n_calls % self.check_freq == 0:
            logger.info(f"Step {self.n_calls}: Training progress checkpoint")
        return True

class MagnetAITrainer:
    """Main training orchestrator for Magnet AI"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['gpu']['device'])
        self.models: Dict[str, PPO] = {}
        self.training_history = []
        
        logger.info(f"🧲 Magnet AI Trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Using {self.config['environment']['num_parallel_envs']} parallel environments")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_env(self, env_id: int) -> HideAndSeekEnv:
        """Create a single environment instance"""
        return HideAndSeekEnv(self.config)
    
    def _create_vec_env(self, num_envs: int):
        """Create vectorized environment for parallel training"""
        env_list = [lambda: self._create_env(i) for i in range(num_envs)]
        return DummyVecEnv(env_list)
    
    def train_student_agent(self, agent_name: str, num_envs: int = 1):
        """Train a student agent using PPO"""
        logger.info(f"\n🎓 Starting training for {agent_name}...")
        
        # Create vectorized environment
        vec_env = self._create_vec_env(num_envs)
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.config['training']['learning_rate'],
            n_steps=self.config['training']['n_steps'],
            batch_size=self.config['training']['batch_size'],
            n_epochs=self.config['training']['n_epochs'],
            gamma=self.config['training']['gamma'],
            gae_lambda=self.config['training']['gae_lambda'],
            clip_range=self.config['training']['clip_range'],
            ent_coef=self.config['training']['ent_coef'],
            vf_coef=self.config['training']['vf_coef'],
            device=self.device,
            verbose=1
        )
        
        # Setup callback
        log_dir = self.config['logging']['log_dir']
        callback = MagnetTrainingCallback(
            check_freq=self.config['logging']['save_interval'],
            log_dir=log_dir
        )
        
        # Train
        total_steps = self.config['training']['total_timesteps']
        logger.info(f"Training {agent_name} for {total_steps:,} steps...")
        
        try:
            model.learn(
                total_timesteps=total_steps,
                callback=callback,
                progress_bar=True
            )
            
            # Save model
            model_path = f"{log_dir}/{agent_name}_final.zip"
            model.save(model_path)
            logger.info(f"✅ {agent_name} saved to {model_path}")
            
            self.models[agent_name] = model
            self.training_history.append({
                'agent': agent_name,
                'timestamp': datetime.now().isoformat(),
                'total_steps': total_steps,
                'status': 'completed'
            })
            
        except KeyboardInterrupt:
            logger.warning(f"⚠️ Training interrupted for {agent_name}")
            self.training_history.append({
                'agent': agent_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'interrupted'
            })
        finally:
            vec_env.close()
    
    def train_all_agents(self):
        """Train all agents: 2 student hiders + 1 snitch detector"""
        agents_to_train = [
            "student_hider_1",
            "student_hider_2",
            "snitch_monitor"
        ]
        
        for agent_name in agents_to_train:
            self.train_student_agent(
                agent_name,
                num_envs=self.config['environment']['num_parallel_envs']
            )
        
        logger.info("\n" + "="*50)
        logger.info("🎉 All agents trained successfully!")
        logger.info(f"Models saved in: {self.config['logging']['log_dir']}")
        logger.info("="*50)
    
    def evaluate_agent(self, agent_name: str, num_episodes: int = 5):
        """Evaluate a trained agent"""
        if agent_name not in self.models:
            logger.error(f"Agent {agent_name} not found in trained models")
            return
        
        logger.info(f"\n📊 Evaluating {agent_name} for {num_episodes} episodes...")
        
        model = self.models[agent_name]
        env = self._create_env(0)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            steps = 0
            done_dict = {i: False for i in range(env.total_agents)}
            
            while not all(done_dict.values()):
                actions = {}
                for agent_id in range(env.total_agents):
                    action, _ = model.predict(obs[agent_id], deterministic=True)
                    actions[agent_id] = action
                
                obs, rewards, done_dict, _ = env.step(actions)
                episode_reward += sum(rewards.values())
                steps += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}, Steps={steps}")
        
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        logger.info(f"\nAverage Reward: {avg_reward:.2f}")
        logger.info(f"Average Episode Length: {avg_length:.0f}")
        
        env.close()

def main():
    """Main training script"""
    import sys
    
    logger.info("\n" + "="*50)
    logger.info("🧲 MAGNET AI - TRAINING SYSTEM")
    logger.info("="*50 + "\n")
    
    trainer = MagnetAITrainer()
    
    # Start training
    trainer.train_all_agents()
    
    # Evaluate if requested
    if "--evaluate" in sys.argv:
        for agent in ["student_hider_1", "student_hider_2"]:
            trainer.evaluate_agent(agent, num_episodes=3)

if __name__ == "__main__":
    main()
