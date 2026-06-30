#!/usr/bin/env python
"""
Magnet AI - Training Data Consolidation
Consolidate and save all learned data from comprehensive training
"""

import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('logs/training_consolidation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def consolidate_training_data():
    """Consolidate and save all learned training data"""
    
    print("\n" + "="*70)
    print("🧲 MAGNET AI - TRAINING DATA CONSOLIDATION")
    print("="*70 + "\n")
    
    # Load comprehensive training session
    session_file = Path('logs/comprehensive_training_session.json')
    if not session_file.exists():
        logger.error("❌ No training session found. Run: python comprehensive_math_training.py")
        return
    
    logger.info("📖 Loading comprehensive training data...")
    with open(session_file, 'r') as f:
        session_data = json.load(f)
    
    # Create consolidated training data
    consolidated = {
        'training_metadata': {
            'consolidation_timestamp': datetime.now().isoformat(),
            'total_problems_trained': session_data['curriculum_size'],
            'total_students': len(session_data['students']),
            'training_sessions': 1,
            'last_updated': datetime.now().isoformat(),
            'version': '2.0'
        },
        'curriculum_overview': {
            'total_problems': 105,
            'categories': {
                'arithmetic': {
                    'addition': 10,
                    'subtraction': 10,
                    'multiplication': 10,
                    'division': 10,
                    'total': 40
                },
                'word_problems': {
                    'subtraction_context': 15,
                    'addition_context': 15,
                    'multiplication_context': 15,
                    'division_context': 15,
                    'total': 60
                },
                'advanced': {
                    'percentages': 15,
                    'complex_multi_step': 15,
                    'total': 30
                }
            }
        },
        'student_performance': [],
        'teacher_performance': session_data['teacher'],
        'aggregate_statistics': session_data['summary'],
        'learning_insights': {
            'hider_strategies': [],
            'seeker_strategies': []
        },
        'shared_memory': {
            'global_insights': [
                'Can perform basic arithmetic (+, -, *, /)'
            ],
            'best_hider_techniques': [],
            'best_seeker_techniques': [],
            'anomaly_patterns': [],
            'last_sync': datetime.now().isoformat(),
            'instances_connected': len(session_data['students'])
        },
        'training_session_details': session_data,
        'episode_history': []
    }
    
    # Add student performance
    for student in session_data['students']:
        consolidated['student_performance'].append({
            'name': student['name'],
            'accuracy': student['accuracy'],
            'facts_learned': student['learned_facts'],
            'total_attempts': student['stats']['total_attempts'],
            'correct_answers': student['stats']['correct'],
            'incorrect_answers': student['stats']['incorrect'],
            'improvement_potential': 100 - student['accuracy']
        })
    
    # Add episode snapshot
    consolidated['episode_history'].append({
        'timestamp': datetime.now().isoformat(),
        'episode': 1,
        'problems_trained': session_data['curriculum_size'],
        'average_accuracy': session_data['summary']['average_accuracy'],
        'average_learned_facts': session_data['summary']['average_learned_facts'],
        'total_attempts': session_data['summary']['total_attempts'],
        'total_correct': session_data['summary']['total_correct']
    })
    
    # Save consolidated data
    logger.info("\n💾 Saving consolidated training data...")
    
    training_data_file = Path('training_data.json')
    with open(training_data_file, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    # Create backup
    backup_file = Path(f"training_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    import shutil
    shutil.copy2(training_data_file, backup_file)
    
    logger.info(f"✅ Main data saved: {training_data_file}")
    logger.info(f"✅ Backup created: {backup_file}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("\n📈 CONSOLIDATION SUMMARY\n")
    
    logger.info(f"""
    🏙️ TRAINING INFRASTRUCTURE:
      - Total Curriculum: {consolidated['curriculum_overview']['total_problems']} problems
      - Categories: 3 (Arithmetic, Word Problems, Advanced)
      - Student Instances: {len(session_data['students'])}
      - Teacher Instances: 1
    
    🧠 LEARNING RESULTS:
      - Average Student Accuracy: {session_data['summary']['average_accuracy']:.1f}%
      - Average Facts Learned: {int(session_data['summary']['average_learned_facts'])}
      - Total Training Interactions: {session_data['teacher']['stats']['total_feedback_given']}
      - Teacher Corrections: {session_data['teacher']['stats']['corrections_made']}
    
    📚 STUDENT PERFORMANCE:
    """)
    
    for student in consolidated['student_performance']:
        logger.info(f"      {student['name']}:")
        logger.info(f"         Accuracy: {student['accuracy']:.1f}%")
        logger.info(f"         Learned: {student['facts_learned']} facts")
        logger.info(f"         Correct: {student['correct_answers']}/{student['total_attempts']}")
    
    logger.info(f"""
    💾 DATA SAVED:
      - Main File: training_data.json
      - Backup: {backup_file.name}
      - Session Log: comprehensive_training_session.json
      - Full Log: comprehensive_math_training.log
    
    ✅ Training data successfully consolidated!
    ✅ Ready for: continuous training, data analysis, model updates
    """)
    
    logger.info("="*70 + "\n")
    
    return consolidated

if __name__ == "__main__":
    try:
        consolidate_training_data()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
