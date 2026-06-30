# 🧲 Magnet AI - Multi-Agent Reinforcement Learning System

A production-grade AI training system featuring multi-agent reinforcement learning with mathematical reasoning capabilities, Claude-like chat interface, live conversation saving, and advanced safety mechanisms.

## 🎯 Features

- **Multi-Agent RL Training**: Train multiple AI agents simultaneously using PPO
- **Mathematical Reasoning Module**: Teach agents to calculate and reason through problems instead of guessing
- **Knowledge Distillation**: Transfer expert mathematical knowledge into agent policies
- **Hide-and-Seek Environment**: 3D game environment for agent learning
- **Safety Mechanisms**: Snitch AI to detect reward hacking and anomalies
- **Claude-like Chat Interface**: Interactive chat with like/dislike feedback
- **Live Conversation Saving**: All conversations auto-saved in real-time
- **Feedback System**: Teacher input, user feedback, and snitch alerts
- **Parallel Training**: Run multiple game instances simultaneously
- **Curriculum Learning**: Problems increase in difficulty as agents learn
- **GPU Optimization**: Designed for RTX 3060/4070 class GPUs

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.9+
# CUDA 11.8+ (for GPU support)
# Git
```

### Installation

```bash
# Clone the repository
git clone https://github.com/daddymai22/magnet-ai.git
cd magnet-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` to adjust:
- GPU settings (device, VRAM limit)
- Training parameters (learning rate, batch size, epochs)
- Environment settings (number of agents, map size, episode length)
- Safety thresholds

```yaml
gpu:
  device: "cuda:0"  # Or "cpu" if no GPU
  vram_limit_mb: 8000

training:
  total_timesteps: 1000000  # Increase for longer training
  learning_rate: 3e-4
  batch_size: 64
  n_epochs: 10

environment:
  num_parallel_envs: 4  # Increase for more simultaneous games
```

## 📚 Usage

### 1. Train Agents (with Math Reasoning)

```bash
python training_with_reasoning.py
```

This will:
- Initialize reasoning layers for student agents
- Pre-train reasoning on easy math problems
- Create 4 parallel game environments
- Train 2 student hider agents + 1 snitch detector
- Interleave RL training with mathematical reasoning
- Save models and reasoning results every checkpoint to `./logs/`
- Display training progress with both RL and reasoning metrics

**Expected Output:**
```
==================================================
🧲 MAGNET AI - TRAINING SYSTEM WITH MATH REASONING
==================================================

Device: cuda:0
Using 4 parallel environments
📐 Loaded 100 math problems for curriculum learning

🎓 Starting training for student_hider_1...
📐 Initializing reasoning layer for student_hider_1...
   Pre-training on 50 problems...
   Pre-training complete! Avg accuracy: 92.45%

Training student_hider_1 for 1,000,000 steps...
[████████████████████████████████████] 1000000/1000000 [100:45<00:00]

📐 Interleaved Math Reasoning Training at step 100000
   Average Loss: 0.0234
   Average Accuracy: 95.67%

✅ student_hider_1 saved to ./logs/student_hider_1_final.zip
✅ Reasoning layer saved to ./logs/student_hider_1_reasoning.pt
✅ Reasoning results saved to ./logs/student_hider_1_reasoning_results.json
```

**Training Duration:**
- 1M steps with 4 parallel envs on RTX 3060: ~3-5 hours (includes reasoning training)
- For faster training, increase `num_parallel_envs` in config (uses more VRAM)

### 2. Train Agents (Original RL only)

For pure RL training without mathematical reasoning:

```bash
python training.py
```

This will use the original training pipeline without the math reasoning module.

### 3. Chat with Trained Agents

```bash
python chat_interface.py
```

**Interactive Commands:**
```
> new                    # Start new conversation
> Type your message      # Chat with agent
> like                   # Like the last response
> dislike                # Dislike the last response
> list                   # List all conversations
> load <conversation_id> # Load saved conversation
> exit                   # Quit
```

**Example Session:**
```
🧲 MAGNET AI - Chat Interface

> new
Conversation title (or press Enter): My First Chat
✨ Started new conversation: My First Chat

> How does hide and seek training work?
👤 You: How does hide and seek training work?

🤖 Agent is thinking...
🧲 Agent: The AI learns through reinforcement learning where...

> like
👍 Feedback recorded!

> list
📚 Your Conversations:
1. My First Chat (2 messages)

> exit
👋 Goodbye!
```

### 4. Evaluate Agents

```bash
python training_with_reasoning.py --evaluate
```

Runs 3 evaluation episodes per agent and displays:
- RL performance (rewards, episode length)
- Math reasoning performance (accuracy on held-out problems)
- Reasoning training statistics

## 📁 Project Structure

```
magnet-ai/
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── training.py                    # Original training script (RL only)
├── training_with_reasoning.py     # Enhanced training with math reasoning
├── environment.py                 # Hide-and-seek game environment
├── chat_interface.py              # Chat UI and conversation manager
├── feedback_system.py             # Feedback tracking and management
├── math_reasoning.py              # Math reasoning module (NEW!)
├── ARCHITECTURE.md                # Detailed architecture documentation
├── logs/                          # Saved models and training logs
├── conversations/                 # Saved conversations (auto-created)
└── README.md
```

## 🏗️ Architecture

### Training Pipeline with Reasoning

```
[Config] → [Environment] → [PPO Agent] → [Training Loop]
                 ↓                            ↓
          [Game Instances]        [Math Reasoning Layer]
            (4 parallel)           (Interleaved training)
                 ↓                            ↓
     [Reward Calculation]    [Knowledge Distillation]
          ↓
     [Snitch Detection]
          ↓
     [Model Update]
          ↓
     [Checkpoint Save]
```

### Multi-Agent System

- **Student Hiders (2)**: Learn to evade seekers using PPO + math reasoning for calculations
- **Snitch Detector (1)**: Monitors for reward hacking and anomalies
- **Feedback Loop**: Teacher input + user feedback shapes training

### Math Reasoning System

The new math reasoning module teaches agents to:

1. **Learn Mathematical Operations**: Addition, multiplication, geometry calculations
2. **Step-by-Step Reasoning**: Break problems into interpretable steps
3. **Curriculum Learning**: Start with easy problems, progress to complex ones
4. **Knowledge Distillation**: Transfer expert knowledge into agent policies

**Reasoning Architecture:**
```
Input Problem
    ↓
[Reasoning Layer - Step 1] → Intermediate Output
    ↓
[Reasoning Layer - Step 2] → Intermediate Output
    ↓
[Reasoning Layer - Step 3] → Intermediate Output
    ↓
[Reasoning Layer - Step 4] → Intermediate Output
    ↓
[Reasoning Layer - Step 5] → Intermediate Output
    ↓
[Final Calculator] → Final Answer
```

**Problem Types:**
- Addition problems (difficulty 1-10)
- Multiplication problems (difficulty 1-10)
- Euclidean distance calculations (for spatial reasoning in hide-and-seek)

### Chat System

```
[User Input] → [Conversation Manager] → [LLM] → [Live Save]
                         ↓
               [Feedback System]
               (like/dislike/teacher)
```

## 🔧 Advanced Configuration

### GPU Optimization

**For RTX 3060 (12GB VRAM):**
```yaml
training:
  batch_size: 64
environment:
  num_parallel_envs: 4
```

**For RTX 4070 (12GB VRAM):**
```yaml
training:
  batch_size: 128
environment:
  num_parallel_envs: 8  # Faster training
```

**For CPU (not recommended):**
```yaml
gpu:
  device: "cpu"
environment:
  num_parallel_envs: 1
  render_fps: 10  # Lower FPS to reduce CPU load
```

### Adjusting Training Length

```yaml
training:
  total_timesteps: 5000000  # 5M for extensive training (12-20 hours)
  # vs
  total_timesteps: 100000   # 100K for quick testing (15 minutes)
```

### Math Reasoning Configuration

Control how often reasoning training is interleaved:

```python
# In training_with_reasoning.py
ReasoningAugmentedCallback(
    check_freq=self.config['logging']['save_interval'],
    log_dir=log_dir,
    math_distiller=distiller,
    math_problems=self.math_problems,
    reasoning_freq=5  # Train math every 5 RL checkpoints (adjust this)
)
```

### Safety Settings

```yaml
safety:
  enable_snitch_model: true
  snitch_threshold: 0.7     # Alert if suspicious > 70%
  reward_hack_detection: true
```

## 📊 Monitoring Training

Training logs are saved to `./logs/`:

```
logs/
├── student_hider_1_final.zip
├── student_hider_1_reasoning.pt          # NEW: Reasoning layer weights
├── student_hider_1_reasoning_results.json # NEW: Math training history
├── student_hider_2_final.zip
├── student_hider_2_reasoning.pt          # NEW
├── student_hider_2_reasoning_results.json # NEW
├── snitch_monitor_final.zip
└── training_log.txt
```

View real-time progress:
```bash
tail -f logs/training_log.txt
```

### Reasoning Results Format

```json
{
  "training_history": [
    {
      "problem_id": "dist_1.5_2.3_4.2_5.1",
      "loss": 0.0234,
      "accuracy": 0.95,
      "error": 0.12,
      "expected": 4.24,
      "predicted": 4.36
    },
    ...
  ],
  "num_trained": 100
}
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size and parallel environments
num_parallel_envs: 2  # Instead of 4
batch_size: 32        # Instead of 64
```

### Slow Training

```bash
# Increase parallel environments (if you have VRAM)
num_parallel_envs: 8  # More parallelism = faster

# Or increase batch size
batch_size: 128  # Process more data per update
```

### Math Reasoning Training Fails

```bash
# Check if math_reasoning.py is in the same directory
ls math_reasoning.py

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"

# Run with verbose logging
LOGLEVEL=DEBUG python training_with_reasoning.py
```

### Chat Not Responding

```bash
# Ensure trained models exist
ls ./logs/*.zip  # Should show student_hider_*.zip files

# Restart chat interface
python chat_interface.py
```

## 🔐 Safety & Ethics

This system includes built-in safety mechanisms:

1. **Snitch AI**: Monitors for reward hacking
2. **Teacher Feedback**: Human guidance shapes training
3. **Reward Clipping**: Maximum reward per step capped
4. **Anomaly Detection**: Flags suspicious behaviors
5. **Math Verification**: Reasoning layer validates calculations

**Important**: This is a research/educational system. Always:
- Monitor training for unexpected behaviors
- Review snitch alerts regularly
- Provide regular teacher feedback
- Keep conversations and feedback logs
- Validate mathematical reasoning outputs

## 📈 Performance Metrics

Expected results after training:

**RL Performance:**
- **Hiders**: Learn to evade for ~200+ steps before capture
- **Seekers**: Catch hiders consistently after 500K+ steps
- **Snitch**: >90% accuracy detecting anomalies
- **Chat**: Responds with contextual awareness within 1-2 seconds

**Math Reasoning Performance:**
- **Addition**: >98% accuracy on problems up to difficulty 10
- **Multiplication**: >96% accuracy on problems up to difficulty 10
- **Geometry**: >90% accuracy on distance calculations
- **Overall**: Improves throughout training with curriculum learning

## 🚦 Roadmap (Tier 2+)

- [ ] Multi-agent communication
- [ ] Live game visualization (100+ windows)
- [ ] Teacher feedback integration with reasoning
- [ ] Advanced safety mechanisms
- [ ] Model fine-tuning on conversation data
- [ ] Web UI dashboard with reasoning visualization
- [ ] Distributed training across multiple GPUs
- [ ] Natural language math problem solving
- [ ] Integration with external reasoning engines
- [ ] Explainable AI for reasoning steps

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request

Areas for contribution:
- Additional math problem types
- Improved reasoning architectures
- Better curriculum learning strategies
- Performance optimizations
- Documentation improvements

## 📧 Support

For issues or questions:
1. Check troubleshooting section
2. Review GitHub issues
3. Create new issue with details

Include when reporting issues:
- Error message (full traceback)
- Your GPU model and VRAM
- Python version
- Steps to reproduce

---

**Built with ❤️ by daddymai22**

*"AI training, done right. With reasoning, not just guessing."*
