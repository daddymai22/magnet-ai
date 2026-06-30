#!/usr/bin/env python
"""
Magnet AI - Continuous Training System
Runs indefinitely, training agents and accumulating learning data
"""

import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import time
import logging
from collections import deque
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousTrainingMonitor:
    """Monitors and tracks continuous training progress"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_steps = deque(maxlen=window_size)
        self.episode_timestamps = deque(maxlen=window_size)
        self.hider_wins = deque(maxlen=window_size)
        self.anomalies_detected = deque(maxlen=window_size)
        
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = datetime.now()
        self.last_update_time = datetime.now()
    
    def record_episode(self, episode_reward: float, episode_steps: int, 
                      hider_won: bool, anomalies: int) -> None:
        """Record episode results"""
        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(episode_steps)
        self.episode_timestamps.append(datetime.now())
        self.hider_wins.append(1 if hider_won else 0)
        self.anomalies_detected.append(anomalies)
        
        self.total_episodes += 1
        self.total_steps += episode_steps
    
    def get_stats(self) -> dict:
        """Get current training statistics"""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'avg_reward_last_100': float(np.mean(self.episode_rewards)),
            'max_reward_last_100': float(np.max(self.episode_rewards)),
            'min_reward_last_100': float(np.min(self.episode_rewards)),
            'avg_steps': float(np.mean(self.episode_steps)),
            'win_rate_last_100': float(np.mean(self.hider_wins)),
            'anomalies_last_100': sum(self.anomalies_detected),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'training_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_progress_report(self) -> str:
        """Generate human-readable progress report"""
        stats = self.get_stats()
        if not stats:
            return "No data yet"
        
        duration = (datetime.now() - self.start_time).total_seconds() / 3600
        
        report = f"""
╔{'═'*68}╗
║ 🧲 MAGNET AI - CONTINUOUS TRAINING REPORT                          ║
╠{'═'*68}╣
║ Episodes Trained:        {stats['total_episodes']:>45} ║
║ Total Steps:             {stats['total_steps']:>45,} ║
║ Training Duration:       {duration:>44.2f}h ║
║                                                                    ║
║ Average Reward (last 100): {stats['avg_reward_last_100']:>36.4f} ║
║ Max Reward (last 100):     {stats['max_reward_last_100']:>36.4f} ║
║ Win Rate (last 100):       {stats['win_rate_last_100']*100:>35.1f}% ║
║ Anomalies Detected:        {sum(self.anomalies_detected):>36} ║
║                                                                    ║
║ Average Episode Length:    {stats['avg_steps']:>35.1f} steps ║
║ Training Speed:            {(stats['total_steps']/duration)/1000:>33.1f}k steps/hour ║
╚{'═'*68}╝
        """
        return report

class ContinuousTrainer:
    """Manages continuous training of AI agents"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        from environment import HideAndSeekEnv
        from training_data_manager import TrainingDataManager
        
        self.env = HideAndSeekEnv(self.config)
        self.data_manager = TrainingDataManager()
        self.monitor = ContinuousTrainingMonitor()
        
        logger.info(f"🧲 Continuous Trainer initialized")
        logger.info(f"   Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
        logger.info(f"   Map Size: {self.env.map_size}x{self.env.map_size}")
    
    def run_episode(self) -> tuple:
        """Run single training episode"""
        obs = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        hiders_survived = False
        anomalies_count = 0
        
        for step in range(self.env.episode_length):
            # Generate actions
            actions = {}
            for agent_id in range(self.env.total_agents):
                action = self.env.action_space.sample()
                actions[agent_id] = action
            
            obs, rewards, dones, info = self.env.step(actions)
            episode_reward += sum(rewards.values())
            episode_steps += 1
            
            # Count anomalies
            for agent_id, agent_info in info.items():
                if agent_info.get('suspicious', False):
                    anomalies_count += 1
            
            # Check if hiders survived
            hiders_alive = sum(1 for a in self.env.agents.values() 
                              if a.agent_type.name == "HIDER" and not a.found)
            if hiders_alive > 0:
                hiders_survived = True
            
            if all(dones.values()):
                break
        
        # Record episode
        self.monitor.record_episode(
            episode_reward / self.env.total_agents,
            episode_steps,
            hiders_survived,
            anomalies_count
        )
        
        return episode_reward, episode_steps, hiders_survived, anomalies_count
    
    def train_continuous(self, duration_hours: float = 24, report_interval: int = 100):
        """Run continuous training for specified duration"""
        logger.info(f"\n🚀 Starting continuous training for {duration_hours} hours...\n")
        
        start_time = time.time()
        duration_seconds = duration_hours * 3600
        episode_count = 0
        
        try:
            while (time.time() - start_time) < duration_seconds:
                episode_reward, steps, hider_won, anomalies = self.run_episode()
                episode_count += 1
                
                # Report progress
                if episode_count % report_interval == 0:
                    elapsed = (time.time() - start_time) / 3600
                    logger.info(f"\n📊 Episode {episode_count} (Time: {elapsed:.2f}h)")
                    logger.info(f"   Last episode reward: {episode_reward:.3f}")
                    logger.info(f"   Last episode steps: {steps}")
                    logger.info(f"   Hiders survived: {'✅ Yes' if hider_won else '❌ No'}")
                    logger.info(f"   Anomalies detected: {anomalies}")
                    logger.info(self.monitor.get_progress_report())
                    
                    # Save progress
                    self._save_progress()
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Training interrupted by user")
        
        finally:
            logger.info(f"\n✅ Continuous training stopped after {episode_count} episodes")
            self._save_final_report(episode_count, time.time() - start_time)
    
    def _save_progress(self) -> None:
        """Save current training progress"""
        stats = self.monitor.get_stats()
        if stats:
            # Update training data
            self.data_manager.data['training_metadata']['total_episodes'] = stats['total_episodes']
            self.data_manager.data['training_metadata']['training_steps_total'] = stats['total_steps']
            
            # Save episode snapshots
            episode_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'episodes': stats['total_episodes'],
                'steps': stats['total_steps'],
                'avg_reward': stats['avg_reward_last_100'],
                'win_rate': stats['win_rate_last_100'],
                'anomalies': stats['anomalies_last_100']
            }
            
            if 'episode_history' not in self.data_manager.data:
                self.data_manager.data['episode_history'] = []
            
            self.data_manager.data['episode_history'].append(episode_snapshot)
            self.data_manager.save_data()
    
    def _save_final_report(self, total_episodes: int, total_seconds: float) -> None:
        """Save final training report"""
        stats = self.monitor.get_stats()
        
        report = {
            'training_session': {
                'start_time': self.monitor.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': total_seconds,
                'total_episodes': total_episodes,
                'total_steps': stats['total_steps']
            },
            'final_statistics': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        report_file = Path('logs/training_session_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📄 Training report saved to {report_file}")
        logger.info(self.monitor.get_progress_report())

def main():
    print("\n" + "="*70)
    print("🧲 MAGNET AI - CONTINUOUS TRAINING SYSTEM")
    print("="*70)
    print("""
This system will train your AI continuously.
Progress will be logged to: logs/continuous_training.log

When you're ready to check progress or update training data,
run: python check_training.py

To stop training: Press Ctrl+C
    """)
    print("="*70 + "\n")
    
    trainer = ContinuousTrainer()
    
    # Get duration from command line or default to 24 hours
    duration = 24.0  # Default: 24 hours
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid duration. Using default 24 hours")
    
    logger.info(f"📌 Training will run for {duration} hours")
    logger.info(f"📌 You can check progress anytime with: python check_training.py")
    logger.info(f"📌 To update data, run: python update_training_data.py\n")
    
    trainer.train_continuous(duration_hours=duration, report_interval=100)

if __name__ == "__main__":
    main()
