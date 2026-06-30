#!/usr/bin/env python
"""
Magnet AI - Extended Math Training
More advanced math problems with deeper learning
"""

import logging
from datetime import datetime
from pathlib import Path
import json
from math_training import TeacherAI, StudentAI, MathQuestion

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('logs/advanced_math_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_extended_math_training():
    """Run extended math training with more problems"""
    
    print("\n" + "="*70)
    print("🧠 MAGNET AI - EXTENDED MATH TRAINING")
    print("="*70 + "\n")
    
    # Initialize
    teacher = TeacherAI(agent_id=999, agent_name="Teacher AI")
    students = [
        StudentAI(agent_id=0, agent_name="Student AI #1"),
        StudentAI(agent_id=1, agent_name="Student AI #2"),
        StudentAI(agent_id=2, agent_name="Student AI #3")
    ]
    
    # Extended math curriculum
    curriculum = [
        # Basic addition
        MathQuestion("5 + 5 = ?", 10),
        MathQuestion("10 + 10 = ?", 20),
        MathQuestion("3 + 7 = ?", 10),
        
        # Basic multiplication
        MathQuestion("5 times 5 = ?", 25),
        MathQuestion("5x5 = ?", 25),
        MathQuestion("2x10 = ?", 20),
        MathQuestion("3 x 4 = ?", 12),
        
        # Harder problems
        MathQuestion("15 + 25 = ?", 40),
        MathQuestion("7 x 8 = ?", 56),
        MathQuestion("100 + 50 = ?", 150),
    ]
    
    logger.info("📚 Curriculum: 10 math problems\n")
    for i, prob in enumerate(curriculum, 1):
        logger.info(f"{i}. {prob.question} -> {prob.correct_answer}")
    
    logger.info("\n" + "="*70)
    logger.info("\n🔄 TRAINING CYCLE 1: Initial Learning\n")
    
    # Training cycle 1
    for idx, problem in enumerate(curriculum, 1):
        logger.info(f"\n[Problem {idx}/{len(curriculum)}] {problem.question}")
        
        for student in students:
            q, answer = student.attempt_question(problem.question)
            is_correct = problem.is_correct(answer)
            
            if not is_correct:
                feedback = teacher.provide_feedback(
                    problem.question,
                    answer,
                    problem.correct_answer,
                    False
                )
                logger.info(f"  {student.agent_name}: {answer} ❌")
                logger.info(f"    {feedback.split(chr(10))[2]}")
                student.learn_from_feedback(problem.question, problem.correct_answer, feedback)
            else:
                logger.info(f"  {student.agent_name}: {answer} ✅")
            
            student.record_attempt(q, answer, is_correct)
    
    # Review after cycle 1
    logger.info("\n" + "="*70)
    logger.info("\n📊 AFTER CYCLE 1 - Performance\n")
    for student in students:
        stats = student.get_stats()
        logger.info(f"{student.agent_name}: {stats['accuracy']:.1f}% accuracy")
    
    # Training cycle 2 - review
    logger.info("\n" + "="*70)
    logger.info("\n🔄 TRAINING CYCLE 2: Review & Reinforcement\n")
    
    for idx, problem in enumerate(curriculum, 1):
        for student in students:
            q, answer = student.attempt_question(problem.question)
            is_correct = problem.is_correct(answer)
            student.record_attempt(q, answer, is_correct)
            
            if is_correct:
                logger.info(f"  {student.agent_name}: {problem.question} = {answer} ✅")
            else:
                logger.info(f"  {student.agent_name}: {problem.question} = {answer} ❌ (should be {problem.correct_answer})")
                student.learn_from_feedback(problem.question, problem.correct_answer, "Corrected!")
    
    # Final stats
    logger.info("\n" + "="*70)
    logger.info("\n📈 FINAL RESULTS\n")
    
    total_accuracy = 0
    for student in students:
        stats = student.get_stats()
        logger.info(f"{student.agent_name}:")
        logger.info(f"  Total Attempts: {stats['total_attempts']}")
        logger.info(f"  Correct: {stats['correct']}")
        logger.info(f"  Incorrect: {stats['incorrect']}")
        logger.info(f"  Accuracy: {stats['accuracy']:.1f}%")
        logger.info(f"  Facts Learned: {stats['learned_facts']}\n")
        total_accuracy += stats['accuracy']
    
    avg_accuracy = total_accuracy / len(students)
    teacher_stats = teacher.get_stats()
    
    logger.info(f"🎯 Average Accuracy: {avg_accuracy:.1f}%")
    logger.info(f"\n🏆 Training Complete!")
    logger.info(f"   Teacher gave {teacher_stats['total_feedback_given']} feedbacks")
    logger.info(f"   Made {teacher_stats['corrections_made']} corrections")

if __name__ == "__main__":
    try:
        run_extended_math_training()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
