# 🧲 Magnet AI - Multi-Agent Reinforcement Learning System

A production-grade AI training system featuring multi-agent reinforcement learning with a Claude-like chat interface, live conversation saving, and advanced safety mechanisms.

## 🎯 Features

- **Multi-Agent RL Training**: Train multiple AI agents simultaneously using PPO
- **Hide-and-Seek Environment**: 3D game environment for agent learning
- **Safety Mechanisms**: Snitch AI to detect reward hacking and anomalies
- **Claude-like Chat Interface**: Interactive chat with like/dislike feedback
- **Live Conversation Saving**: All conversations auto-saved in real-time
- **Feedback System**: Teacher input, user feedback, and snitch alerts
- **Parallel Training**: Run multiple game instances simultaneously
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

### 1. Train Agents

```bash
python training.py
```

This will:
- Create 4 parallel game environments
- Train 2 student hider agents + 1 snitch detector
- Save models every 100 episodes to `./logs/`
- Display training progress

**Expected Output:**
```
==================================================
🧲 MAGNET AI - TRAINING SYSTEM
==================================================

Device: cuda:0
Using 4 parallel environments

🎓 Starting training for student_hider_1...
Training student_hider_1 for 1,000,000 steps...
[████████████████████████████████████] 1000000/1000000 [100:45<00:00]
✅ student_hider_1 saved to ./logs/student_hider_1_final.zip
```

**Training Duration:**
- 1M steps with 4 parallel envs on RTX 3060: ~2-4 hours
- For faster training, increase `num_parallel_envs` in config (uses more VRAM)

### 2. Chat with Trained Agents

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

### 3. Evaluate Agents

```bash
python training.py --evaluate
```

Runs 3 evaluation episodes per agent and displays metrics.

## 📁 Project Structure

```
magnet-ai/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── training.py              # Main training script
├── environment.py           # Hide-and-seek game environment
├── chat_interface.py        # Chat UI and conversation manager
├── feedback_system.py       # Feedback tracking and management
├── logs/                    # Saved models and training logs
├── conversations/           # Saved conversations (auto-created)
└── README.md
```

## 🏗️ Architecture

### Training Pipeline

```
[Config] → [Environment] → [PPO Agent] → [Training Loop]
                ↓
         [Game Instances]
           (4 parallel)
                ↓
    [Reward Calculation]
         ↓
    [Snitch Detection]
         ↓
    [Model Update]
         ↓
    [Checkpoint Save]
```

### Multi-Agent System

- **Student Hiders (2)**: Learn to evade seekers using PPO
- **Snitch Detector (1)**: Monitors for reward hacking and anomalies
- **Feedback Loop**: Teacher input + user feedback shapes training

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
  total_timesteps: 5000000  # 5M for extensive training (8-12 hours)
  # vs
  total_timesteps: 100000   # 100K for quick testing (10 minutes)
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
├── student_hider_2_final.zip
├── snitch_monitor_final.zip
└── training_log.txt
```

View real-time progress:
```bash
tail -f logs/training_log.txt
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
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

**Important**: This is a research/educational system. Always:
- Monitor training for unexpected behaviors
- Review snitch alerts regularly
- Provide regular teacher feedback
- Keep conversations and feedback logs

## 📈 Performance Metrics

Expected results after training:

- **Hiders**: Learn to evade for ~200+ steps before capture
- **Seekers**: Catch hiders consistently after 500K+ steps
- **Snitch**: >90% accuracy detecting anomalies
- **Chat**: Responds with contextual awareness within 1-2 seconds

## 🚦 Roadmap (Tier 2+)

- [ ] Multi-agent communication
- [ ] Live game visualization (100+ windows)
- [ ] Teacher feedback integration
- [ ] Advanced safety mechanisms
- [ ] Model fine-tuning on conversation data
- [ ] Web UI dashboard
- [ ] Distributed training

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request

## 📧 Support

For issues or questions:
1. Check troubleshooting section
2. Review GitHub issues
3. Create new issue with details

---

**Built with ❤️ by daddymai22**

*"AI training, done right."*
