"""
Integrated training script combining RL with mathematical reasoning.
Agents learn both through reinforcement learning AND explicit knowledge transfer.
"""

import yaml
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from environment import HideAndSeekEnv
from math_reasoning import (
    MathReasoningLayer, 
    KnowledgeDistiller, 
    MathProblemGenerator,
    ReasoningAugmentedAgent,
    save_training_results
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningAugmentedCallback(BaseCallback):
    """Custom callback that integrates math reasoning with RL training"""
    
    def __init__(
        self, 
        check_freq: int, 
        log_dir: str,
        math_distiller: KnowledgeDistiller,
        math_problems: List,
        reasoning_freq: int = 10,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.math_distiller = math_distiller
        self.math_problems = math_problems
        self.reasoning_freq = reasoning_freq
        Path(log_dir).mkdir(exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.math_training_history = []
        
    def _on_step(self) -> bool:
        """Called after every environment step"""
        
        # Regular checkpointing
        if self.n_calls % self.check_freq == 0:
            logger.info(f"Step {self.n_calls}: Training progress checkpoint")
        
        # Interleaved math reasoning training
        if self.n_calls % (self.check_freq * self.reasoning_freq) == 0:
            self._train_reasoning()
        
        return True
    
    def _train_reasoning(self):
        """Train the reasoning layer on math problems"""
        logger.info(f"\n📐 Interleaved Math Reasoning Training at step {self.n_calls}")
        
        # Sample a batch of problems
        problem_batch = np.random.choice(
            self.math_problems, 
            size=min(10, len(self.math_problems)),
            replace=False
        )
        
        # Train on batch
        batch_metrics = self.math_distiller.teach_batch(list(problem_batch))
        
        logger.info(f"   Average Loss: {batch_metrics['avg_loss']:.4f}")
        logger.info(f"   Average Accuracy: {batch_metrics['avg_accuracy']:.2%}")
        
        self.math_training_history.append({
            'step': self.n_calls,
            'metrics': batch_metrics
        })


class MagnetAITrainerWithReasoning:
    """Enhanced training orchestrator combining RL + math reasoning"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['gpu']['device'])
        self.models: Dict[str, PPO] = {}
        self.training_history = []
        self.reasoning_layers: Dict[str, MathReasoningLayer] = {}
        self.distillers: Dict[str, KnowledgeDistiller] = {}
        
        # Generate math curriculum
        self.math_problems = MathProblemGenerator.generate_curriculum(num_problems=100)
        
        logger.info(f"🧲 Magnet AI Trainer with Math Reasoning initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Using {self.config['environment']['num_parallel_envs']} parallel environments")
        logger.info(f"📐 Loaded {len(self.math_problems)} math problems for curriculum learning")
    
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
    
    def _initialize_reasoning_layer(self, agent_name: str):
        """Initialize reasoning layer for an agent"""
        logger.info(f"\n📐 Initializing reasoning layer for {agent_name}...")
        
        # Create reasoning layer
        reasoning_layer = MathReasoningLayer(
            input_size=4,  # Standard math problem input (e.g., x1, y1, x2, y2)
            hidden_size=256,
            num_steps=5
        )
        reasoning_layer = reasoning_layer.to(self.device)
        
        # Create knowledge distiller
        distiller = KnowledgeDistiller(reasoning_layer, learning_rate=1e-4)
        
        # Pre-train on easier problems
        logger.info(f"   Pre-training on {len(self.math_problems)//2} problems...")
        easy_problems = self.math_problems[:len(self.math_problems)//2]
        
        for problem in easy_problems:
            distiller.teach_from_problem(problem)
        
        logger.info(f"   Pre-training complete! Avg accuracy: {np.mean([p['accuracy'] for p in distiller.training_history[-10:]]):.2%}")
        
        self.reasoning_layers[agent_name] = reasoning_layer
        self.distillers[agent_name] = distiller
        
        return reasoning_layer, distiller
    
    def train_student_agent_with_reasoning(
        self, 
        agent_name: str, 
        num_envs: int = 1,
        use_reasoning: bool = True
    ):
        """Train a student agent using PPO + Math Reasoning"""
        logger.info(f"\n🎓 Starting training for {agent_name}...")
        
        # Initialize reasoning if enabled
        reasoning_layer = None
        distiller = None
        if use_reasoning:
            reasoning_layer, distiller = self._initialize_reasoning_layer(agent_name)
        
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
        
        # Setup callback with reasoning
        log_dir = self.config['logging']['log_dir']
        callback = ReasoningAugmentedCallback(
            check_freq=self.config['logging']['save_interval'],
            log_dir=log_dir,
            math_distiller=distiller,
            math_problems=self.math_problems,
            reasoning_freq=5  # Train reasoning every 5 RL checkpoints
        ) if use_reasoning else None
        
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
            
            # Save reasoning layer if used
            if reasoning_layer is not None:
                reasoning_path = f"{log_dir}/{agent_name}_reasoning.pt"
                torch.save(reasoning_layer.state_dict(), reasoning_path)
                logger.info(f"✅ Reasoning layer saved to {reasoning_path}")
                
                # Save training results
                results_path = f"{log_dir}/{agent_name}_reasoning_results.json"
                save_training_results(distiller, results_path)
                logger.info(f"✅ Reasoning results saved to {results_path}")
            
            self.models[agent_name] = model
            self.training_history.append({
                'agent': agent_name,
                'timestamp': datetime.now().isoformat(),
                'total_steps': total_steps,
                'status': 'completed',
                'reasoning_enabled': use_reasoning,
                'math_accuracy': np.mean([p['accuracy'] for p in distiller.training_history]) if distiller else 0.0
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
    
    def train_all_agents_with_reasoning(self):
        """Train all agents: 2 student hiders + 1 snitch detector with math reasoning"""
        agents_to_train = [
            ("student_hider_1", True),   # Use reasoning
            ("student_hider_2", True),   # Use reasoning
            ("snitch_monitor", False)    # No reasoning needed for anomaly detection
        ]
        
        for agent_name, use_reasoning in agents_to_train:
            self.train_student_agent_with_reasoning(
                agent_name,
                num_envs=self.config['environment']['num_parallel_envs'],
                use_reasoning=use_reasoning
            )
        
        logger.info("\n" + "="*50)
        logger.info("🎉 All agents trained with knowledge integration!")
        logger.info(f"Models saved in: {self.config['logging']['log_dir']}")
        logger.info("="*50)
    
    def evaluate_agent_with_reasoning(
        self, 
        agent_name: str, 
        num_episodes: int = 5
    ):
        """Evaluate a trained agent with reasoning capability"""
        if agent_name not in self.models:
            logger.error(f"Agent {agent_name} not found in trained models")
            return
        
        logger.info(f"\n📊 Evaluating {agent_name} for {num_episodes} episodes...")
        
        model = self.models[agent_name]
        env = self._create_env(0)
        
        episode_rewards = []
        episode_lengths = []
        
        # Evaluate RL performance
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
        
        logger.info(f"\n📊 RL Performance:")
        logger.info(f"   Average Reward: {avg_reward:.2f}")
        logger.info(f"   Average Episode Length: {avg_length:.0f}")
        
        # Evaluate reasoning performance if available
        if agent_name in self.distillers:
            distiller = self.distillers[agent_name]
            recent_accuracy = np.mean([p['accuracy'] for p in distiller.training_history[-20:]])
            logger.info(f"\n📐 Math Reasoning Performance:")
            logger.info(f"   Recent Accuracy: {recent_accuracy:.2%}")
            logger.info(f"   Total Problems Trained: {len(distiller.training_history)}")
        
        env.close()
    
    def get_training_summary(self) -> Dict:
        """Generate comprehensive training summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_agents_trained': len(self.training_history),
            'agents': self.training_history,
            'reasoning_layers_created': len(self.reasoning_layers),
            'math_problems_used': len(self.math_problems),
            'device': str(self.device)
        }


def main():
    """Main training script with integrated reasoning"""
    import sys
    
    logger.info("\n" + "="*50)
    logger.info("🧲 MAGNET AI - TRAINING SYSTEM WITH MATH REASONING")
    logger.info("="*50 + "\n")
    
    trainer = MagnetAITrainerWithReasoning()
    
    # Start training with reasoning
    trainer.train_all_agents_with_reasoning()
    
    # Evaluate if requested
    if "--evaluate" in sys.argv:
        logger.info("\n" + "="*50)
        logger.info("📊 EVALUATION PHASE")
        logger.info("="*50 + "\n")
        
        for agent in ["student_hider_1", "student_hider_2", "snitch_monitor"]:
            trainer.evaluate_agent_with_reasoning(agent, num_episodes=3)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("📈 TRAINING SUMMARY")
    logger.info("="*50)
    summary = trainer.get_training_summary()
    for key, value in summary.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()
