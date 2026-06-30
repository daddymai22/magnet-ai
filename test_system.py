#!/usr/bin/env python
"""
Magnet AI - System Test Suite
Verifies all components work correctly before full training
"""

import sys
import torch
import numpy as np
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_header(text):
    print(f"\n{'='*60}")
    print(f"🧲 {text}")
    print(f"{'='*60}\n")

def test_pytorch_gpu():
    """Test PyTorch and GPU availability"""
    print_header("TEST 1: PyTorch & GPU")
    
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Test GPU computation
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        logger.info(f"✅ GPU computation test passed (shape: {z.shape})")
    else:
        logger.warning("⚠️  CUDA not available - will use CPU (much slower)")
    
    return True

def test_environment():
    """Test hide-and-seek environment"""
    print_header("TEST 2: Hide & Seek Environment")
    
    try:
        import yaml
        from environment import HideAndSeekEnv, AgentType
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Creating environment...")
        env = HideAndSeekEnv(config)
        logger.info(f"✅ Environment created")
        logger.info(f"   - Map size: {env.map_size}x{env.map_size}")
        logger.info(f"   - Episode length: {env.episode_length} steps")
        logger.info(f"   - Total agents: {env.total_agents}")
        
        # Reset environment
        logger.info("Resetting environment...")
        obs = env.reset()
        logger.info(f"✅ Environment reset")
        logger.info(f"   - Number of agents: {len(obs)}")
        logger.info(f"   - Observation shape: {obs[0].shape}")
        
        # Run test episode (50 steps)
        logger.info("Running 50 test steps...")
        total_reward = {i: 0 for i in range(env.total_agents)}
        
        for step in range(50):
            # Random actions
            actions = {i: env.action_space.sample() for i in range(env.total_agents)}
            obs, rewards, dones, info = env.step(actions)
            
            for agent_id, reward in rewards.items():
                total_reward[agent_id] += reward
            
            if step % 10 == 0:
                logger.info(f"  Step {step}: rewards = {[f'{r:.2f}' for r in rewards.values()]}")
        
        logger.info(f"✅ Test episode complete")
        logger.info(f"   - Total rewards: {total_reward}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_model():
    """Test PPO model creation and forward pass"""
    print_header("TEST 3: PPO Model & Training")
    
    try:
        import yaml
        from environment import HideAndSeekEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Creating single environment...")
        def make_env():
            return HideAndSeekEnv(config)
        
        vec_env = DummyVecEnv([make_env])
        logger.info(f"✅ Vectorized environment created")
        
        logger.info("Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            verbose=0,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        logger.info(f"✅ PPO model created")
        logger.info(f"   - Policy network: {model.policy}")
        
        logger.info("Running 100 training steps...")
        start_time = time.time()
        model.learn(total_timesteps=100, log_interval=10)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Training completed")
        logger.info(f"   - Time: {elapsed:.2f}s")
        logger.info(f"   - Speed: {100/elapsed:.0f} steps/sec")
        
        # Test prediction
        logger.info("Testing model prediction...")
        obs, _ = vec_env.reset()
        action, _state = model.predict(obs, deterministic=True)
        logger.info(f"✅ Prediction successful (action: {action})")
        
        vec_env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ PPO model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feedback_system():
    """Test feedback system"""
    print_header("TEST 4: Feedback System")
    
    try:
        from feedback_system import FeedbackManager, Feedback, FeedbackType
        from datetime import datetime
        
        logger.info("Creating feedback manager...")
        fm = FeedbackManager()
        logger.info(f"✅ Feedback manager created")
        
        # Create session
        logger.info("Creating feedback session...")
        fm.create_session("test_session_1", agent_id=0)
        logger.info(f"✅ Session created")
        
        # Add feedbacks
        logger.info("Adding feedback entries...")
        fm.like_response("test_session_1", 0, "Great response!")
        fm.dislike_response("test_session_1", 0, "Not helpful")
        fm.teacher_input("test_session_1", 0, "Student should focus on strategy")
        fm.snitch_alert("test_session_1", 0, "Suspicious wall clipping detected", 0.8)
        logger.info(f"✅ Feedback entries added")
        
        # Get stats
        stats = fm.get_agent_stats(0)
        logger.info(f"✅ Agent stats retrieved")
        logger.info(f"   - Sessions: {stats['sessions']}")
        logger.info(f"   - Positive feedback: {stats['total_positive_feedback']}")
        logger.info(f"   - Negative feedback: {stats['total_negative_feedback']}")
        logger.info(f"   - Feedback ratio: {stats['feedback_ratio']:.2f}")
        logger.info(f"   - Teacher inputs: {stats['teacher_inputs_received']}")
        logger.info(f"   - Snitch alerts: {stats['snitch_alerts']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feedback system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_interface():
    """Test chat interface and persistence"""
    print_header("TEST 5: Chat Interface & Persistence")
    
    try:
        from chat_interface import ChatUI, Message, MessageRole
        from datetime import datetime
        import asyncio
        
        logger.info("Creating chat UI...")
        chat_ui = ChatUI(agent_id=0)
        logger.info(f"✅ Chat UI created")
        
        async def test_chat():
            logger.info("Starting new conversation...")
            await chat_ui.start_new_conversation("Test Conversation")
            logger.info(f"✅ Conversation started")
            
            logger.info("Sending test message...")
            await chat_ui.send_message("Hello, how are you?")
            logger.info(f"✅ Message sent and saved")
            
            logger.info("Adding feedback...")
            await chat_ui.add_feedback("like")
            logger.info(f"✅ Feedback recorded")
            
            logger.info("Listing conversations...")
            chat_ui.list_conversations()
            logger.info(f"✅ Conversations listed")
        
        asyncio.run(test_chat())
        return True
        
    except Exception as e:
        logger.error(f"❌ Chat interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_short_training():
    """Run a short training session"""
    print_header("TEST 6: Short Training Run")
    
    try:
        import yaml
        from training import MagnetAITrainer
        
        logger.info("Creating trainer...")
        trainer = MagnetAITrainer()
        logger.info(f"✅ Trainer created")
        
        logger.info("Running SHORT training (1000 steps only for testing)...")
        logger.info("This will take ~30-60 seconds...")
        
        # Temporarily reduce timesteps
        trainer.config['training']['total_timesteps'] = 1000
        trainer.config['environment']['num_parallel_envs'] = 2
        
        start_time = time.time()
        trainer.train_student_agent("test_student_hider", num_envs=2)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"   - Time: {elapsed:.1f}s")
        logger.info(f"   - Average speed: {1000/elapsed:.0f} steps/sec")
        
        # Check if model was saved
        model_path = Path(f"logs/test_student_hider_final.zip")
        if model_path.exists():
            logger.info(f"✅ Model saved: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧲 MAGNET AI - SYSTEM TEST SUITE")
    print("="*60)
    
    tests = [
        ("PyTorch & GPU", test_pytorch_gpu),
        ("Environment", test_environment),
        ("PPO Model", test_ppo_model),
        ("Feedback System", test_feedback_system),
        ("Chat Interface", test_chat_interface),
        ("Short Training", test_short_training),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for training.\n")
        print("Next steps:")
        print("  1. Adjust config.yaml if needed")
        print("  2. Run: python training.py")
        print("  3. After training, run: python chat_interface.py\n")
        return 0
    else:
        print("⚠️  Some tests failed. Check errors above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
