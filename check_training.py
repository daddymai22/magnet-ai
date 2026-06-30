#!/usr/bin/env python
"""
Magnet AI - Training Progress Checker
View current training statistics without interrupting training
"""

import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_training_progress():
    """Check current training progress"""
    
    print("\n" + "="*70)
    print("🧲 MAGNET AI - TRAINING PROGRESS")
    print("="*70 + "\n")
    
    try:
        # Load training data
        data_file = Path('training_data.json')
        if not data_file.exists():
            logger.error("❌ No training data found. Start training with: python train_continuous.py")
            return
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Display training metadata
        metadata = data.get('training_metadata', {})
        logger.info(f"📊 TRAINING METADATA")
        logger.info(f"   Total Episodes: {metadata.get('total_episodes', 0):,}")
        logger.info(f"   Total Steps: {metadata.get('training_steps_total', 0):,}")
        logger.info(f"   Last Updated: {metadata.get('last_updated', 'Never')}")
        logger.info(f"   Model Instances: {metadata.get('model_instances', 1)}")
        
        # Display agent performance
        logger.info(f"\n🎯 AGENT PERFORMANCE")
        agents = data.get('agent_performance', {})
        for agent_name, perf in agents.items():
            logger.info(f"\n   {agent_name}:")
            logger.info(f"      Episodes: {perf.get('total_episodes', 0):,}")
            logger.info(f"      Avg Survival Steps: {perf.get('average_survival_steps', 0):.1f}")
            logger.info(f"      Win Rate: {perf.get('win_rate', 0)*100:.1f}%")
            logger.info(f"      Contribution to Memory: {perf.get('contribution_to_shared_memory', 0)}")
        
        # Display learned strategies
        logger.info(f"\n💡 LEARNED STRATEGIES")
        insights = data.get('learning_insights', {})
        
        hider_strategies = insights.get('hider_strategies', [])
        if hider_strategies:
            logger.info(f"\n   Hider Strategies ({len(hider_strategies)}):")
            for i, strategy in enumerate(hider_strategies[:5], 1):  # Show top 5
                logger.info(f"      {i}. {strategy.get('strategy_id', 'Unknown')}")
                logger.info(f"         Description: {strategy.get('description', 'N/A')}")
                logger.info(f"         Effectiveness: {strategy.get('effectiveness', 0):.3f}")
        
        seeker_strategies = insights.get('seeker_strategies', [])
        if seeker_strategies:
            logger.info(f"\n   Seeker Strategies ({len(seeker_strategies)}):")
            for i, strategy in enumerate(seeker_strategies[:5], 1):
                logger.info(f"      {i}. {strategy.get('strategy_id', 'Unknown')}")
                logger.info(f"         Effectiveness: {strategy.get('effectiveness', 0):.3f}")
        
        # Display shared memory stats
        logger.info(f"\n🔄 SHARED MEMORY STATUS")
        shared_mem = data.get('shared_memory', {})
        logger.info(f"   Global Insights: {len(shared_mem.get('global_insights', []))}")
        logger.info(f"   Best Hider Techniques: {len(shared_mem.get('best_hider_techniques', []))}")
        logger.info(f"   Best Seeker Techniques: {len(shared_mem.get('best_seeker_techniques', []))}")
        logger.info(f"   Anomaly Patterns Detected: {len(shared_mem.get('anomaly_patterns', []))}")
        logger.info(f"   Last Sync: {shared_mem.get('last_sync', 'Never')}")
        logger.info(f"   Connected Instances: {shared_mem.get('instances_connected', 1)}")
        
        # Show episode history if available
        if 'episode_history' in data:
            history = data['episode_history']
            if history:
                logger.info(f"\n📈 RECENT PROGRESS (last {min(5, len(history))} checkpoints)")
                for record in history[-5:]:
                    logger.info(f"\n   {record.get('timestamp', 'Unknown')}")
                    logger.info(f"      Episodes: {record.get('episodes', 0):,}")
                    logger.info(f"      Steps: {record.get('steps', 0):,}")
                    logger.info(f"      Avg Reward: {record.get('avg_reward', 0):.4f}")
                    logger.info(f"      Win Rate: {record.get('win_rate', 0)*100:.1f}%")
        
        logger.info(f"\n" + "="*70)
        logger.info("✅ Training is running continuously...")
        logger.info("📝 To update training data, run: python update_training_data.py")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Error reading training data: {e}")

if __name__ == "__main__":
    check_training_progress()
