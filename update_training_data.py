#!/usr/bin/env python
"""
Magnet AI - Training Data Update
Update and consolidate training data with new insights
"""

import json
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('logs/data_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_training_data():
    """Update and consolidate training data"""
    
    print("\n" + "="*70)
    print("🧲 MAGNET AI - TRAINING DATA UPDATE")
    print("="*70 + "\n")
    
    data_file = Path('training_data.json')
    backup_file = Path(f"training_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    if not data_file.exists():
        logger.error("❌ No training data found")
        return
    
    # Create backup
    logger.info("📋 Creating backup...")
    import shutil
    shutil.copy2(data_file, backup_file)
    logger.info(f"✅ Backup created: {backup_file}")
    
    # Load current data
    logger.info("\n📖 Loading training data...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Analyze and update
    logger.info("\n🔍 Analyzing training data...")
    
    metadata = data.get('training_metadata', {})
    total_episodes = metadata.get('total_episodes', 0)
    total_steps = metadata.get('total_steps', 0)
    
    logger.info(f"   Total Episodes Trained: {total_episodes:,}")
    logger.info(f"   Total Steps: {total_steps:,}")
    
    # Consolidate strategies
    logger.info("\n🧠 Consolidating learned strategies...")
    
    hider_strategies = data.get('learning_insights', {}).get('hider_strategies', [])
    seeker_strategies = data.get('learning_insights', {}).get('seeker_strategies', [])
    
    # Find best strategies
    if hider_strategies:
        best_hider = max(hider_strategies, key=lambda x: x.get('effectiveness', 0))
        logger.info(f"   Best Hider Strategy: {best_hider.get('strategy_id', 'Unknown')}")
        logger.info(f"      Effectiveness: {best_hider.get('effectiveness', 0):.3f}")
        logger.info(f"      Description: {best_hider.get('description', 'N/A')}")
    
    if seeker_strategies:
        best_seeker = max(seeker_strategies, key=lambda x: x.get('effectiveness', 0))
        logger.info(f"   Best Seeker Strategy: {best_seeker.get('strategy_id', 'Unknown')}")
        logger.info(f"      Effectiveness: {best_seeker.get('effectiveness', 0):.3f}")
    
    # Update shared memory with best techniques
    logger.info("\n📤 Updating shared memory...")
    
    shared_mem = data.get('shared_memory', {})
    
    # Add best techniques
    if hider_strategies:
        top_hider_techs = sorted(hider_strategies, key=lambda x: x.get('effectiveness', 0), reverse=True)[:3]
        for tech in top_hider_techs:
            tech_dict = {
                'strategy_id': tech.get('strategy_id', 'unknown'),
                'description': tech.get('description', ''),
                'effectiveness': tech.get('effectiveness', 0),
                'added_at': datetime.now().isoformat()
            }
            if tech_dict not in shared_mem.get('best_hider_techniques', []):
                if 'best_hider_techniques' not in shared_mem:
                    shared_mem['best_hider_techniques'] = []
                shared_mem['best_hider_techniques'].append(tech_dict)
    
    if seeker_strategies:
        top_seeker_techs = sorted(seeker_strategies, key=lambda x: x.get('effectiveness', 0), reverse=True)[:3]
        for tech in top_seeker_techs:
            tech_dict = {
                'strategy_id': tech.get('strategy_id', 'unknown'),
                'effectiveness': tech.get('effectiveness', 0),
                'added_at': datetime.now().isoformat()
            }
            if tech_dict not in shared_mem.get('best_seeker_techniques', []):
                if 'best_seeker_techniques' not in shared_mem:
                    shared_mem['best_seeker_techniques'] = []
                shared_mem['best_seeker_techniques'].append(tech_dict)
    
    # Update sync time
    shared_mem['last_sync'] = datetime.now().isoformat()
    
    # Calculate aggregate stats
    logger.info("\n📊 Computing aggregate statistics...")
    
    agents = data.get('agent_performance', {})
    total_wins = sum(ag.get('win_rate', 0) * ag.get('total_episodes', 0) for ag in agents.values())
    avg_win_rate = total_wins / max(sum(ag.get('total_episodes', 0) for ag in agents.values()), 1)
    
    aggregate_stats = {
        'timestamp': datetime.now().isoformat(),
        'total_agents': len(agents),
        'average_win_rate': avg_win_rate,
        'total_strategies_discovered': len(hider_strategies) + len(seeker_strategies),
        'shared_knowledge_items': len(shared_mem.get('global_insights', []))
    }
    
    if 'aggregate_statistics' not in data:
        data['aggregate_statistics'] = []
    data['aggregate_statistics'].append(aggregate_stats)
    
    logger.info(f"   Average Win Rate: {avg_win_rate*100:.1f}%")
    logger.info(f"   Total Strategies: {aggregate_stats['total_strategies_discovered']}")
    logger.info(f"   Shared Knowledge Items: {aggregate_stats['shared_knowledge_items']}")
    
    # Save updated data
    logger.info("\n💾 Saving updated training data...")
    data['training_metadata']['last_updated'] = datetime.now().isoformat()
    data['shared_memory'] = shared_mem
    
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"✅ Training data updated")
    
    # Generate summary report
    logger.info("\n" + "="*70)
    logger.info("📄 UPDATE SUMMARY")
    logger.info("="*70)
    
    logger.info(f"""
    Episodes Trained:          {total_episodes:,}
    Total Training Steps:      {total_steps:,}
    Agents Active:             {len(agents)}
    Strategies Discovered:     {aggregate_stats['total_strategies_discovered']}
    Average Win Rate:          {avg_win_rate*100:.1f}%
    Shared Memory Items:       {aggregate_stats['shared_knowledge_items']}
    
    ✅ Training data consolidated and updated
    📋 Backup saved: {backup_file}
    """)
    
    logger.info("="*70 + "\n")

if __name__ == "__main__":
    update_training_data()
