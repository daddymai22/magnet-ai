# Magnet AI - System Architecture

## 🏗️ High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MAGNET AI SYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Training    │    │    Chat      │    │  Feedback    │ │
│  │   System     │───→│  Interface   │←───│   System     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         ↓                   ↓                     ↑        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Environment (Hide & Seek Game)              │  │
│  │  - 4 Parallel Game Instances                        │  │
│  │  - Physics Simulation                               │  │
│  │  - Collision Detection                              │  │
│  │  - Reward Calculation                               │  │
│  └──────────────────────────────────────────────────────┘  │
│         ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      RL Training Loop (PPO - Proximal Policy)        │  │
│  │  - Policy Gradient Updates                          │  │
│  │  - Value Function Training                          │  │
│  │  - Experience Replay                                │  │
│  └──────────────────────────────────────────────────────┘  │
│         ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Safety Layer (Snitch Detector)              │  │
│  │  - Anomaly Detection                                │  │
│  │  - Reward Hack Prevention                           │  │
│  │  - Behavior Monitoring                              │  │
│  └──────────────────────────────────────────────────────┘  │
│         ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Model Checkpoint Storage                 │  │
│  │  - Periodic Model Saves                             │  │
│  │  - Conversation History                             │  │
│  │  - Training Logs                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Component Details

### 1. Training System (`training.py`)

**Purpose**: Orchestrate multi-agent RL training

**Key Classes**:
- `MagnetAITrainer`: Main training coordinator
- `MagnetTrainingCallback`: Progress tracking

**Workflow**:
```
Config Load → Env Creation → PPO Initialization → Training Loop
     ↓
  GPU Setup
     ↓
  Vectorized Environments (4 parallel)
     ↓
  Experience Collection
     ↓
  Policy Updates (every 2048 steps)
     ↓
  Model Checkpoint (every 100 episodes)
     ↓
  Repeat until 1M steps
```

**Computational Flow**:
```
For each update cycle:
  1. Collect 2048 experiences from 4 parallel envs
  2. Compute advantages using GAE
  3. Update policy network (10 epochs)
  4. Update value network
  5. Clip gradients (prevent exploding gradients)
  6. Apply entropy bonus (encourage exploration)
  7. Save checkpoint if needed
```

### 2. Environment (`environment.py`)

**Purpose**: Simulate hide-and-seek game with physics

**Key Classes**:
- `Agent`: Individual agent in environment
- `HideAndSeekEnv`: Main Gym environment
- `AgentType`: Enum (HIDER, SEEKER, SNITCH)

**Observation Space** (per agent):
```
[
  self.x, self.y,                          # Position (2)
  self.vx, self.vy,                        # Velocity (2)
  found_status,                            # Found (1)
  rel_x_agent1, rel_y_agent1, dist_1,     # Other agents (3 per)
  rel_x_agent2, rel_y_agent2, dist_2,
  ...,
  rel_x_snitch, rel_y_snitch, dist_snitch # Snitch (3)
]
```
Total: 5 + (3 * num_agents) = ~17 dimensions

**Action Space** (per agent):
```
[
  forward/backward (-1 to 1),    # X velocity
  left/right (-1 to 1),          # Y velocity
  jump (0 or 1)                  # Unused for now
]
```

**Reward Structure**:
```
Hiders:
  +0.1 per step (survival bonus)
  -1.0 when caught
  
Seekers:
  +0.5 per hider caught (scales with time)
  
Snitch:
  Monitors for anomalies
  Penalizes suspicious behavior
```

**Physics Simulation**:
```python
# Velocity update with friction
v_new = v_old * 0.95 + a * dt

# Speed clamping
if speed > max_speed:
  v = (v / speed) * max_speed

# Position update
x_new = x_old + v * dt

# Boundary clipping
x = clamp(x, -map_size/2, map_size/2)
```

### 3. Chat Interface (`chat_interface.py`)

**Purpose**: Interactive chat with live persistence

**Key Classes**:
- `Message`: Single message with metadata
- `Conversation`: Chat session with messages
- `ConversationManager`: Handles persistence
- `SimpleLLMChat`: LLM interface (placeholder)
- `ChatUI`: CLI interface

**Data Flow**:
```
User Input
    ↓
Add to Conversation (in-memory)
    ↓
Live Save to JSON (async)
    ↓
Generate Response (LLM)
    ↓
Add Response to Conversation
    ↓
Live Save to JSON
    ↓
Display to User
```

**Conversation Persistence**:
```
conversations/
├── conv_0_1720000000.json  # Agent 0, timestamp
├── conv_0_1720001000.json
└── conv_1_1720002000.json  # Agent 1
```

Each JSON file:
```json
{
  "conversation_id": "conv_0_1720000000",
  "agent_id": 0,
  "title": "My First Chat",
  "created_at": "2024-06-30T10:00:00",
  "messages": [
    {
      "role": "user",
      "content": "Hello!",
      "timestamp": "2024-06-30T10:00:01",
      "feedback": null
    },
    {
      "role": "assistant",
      "content": "Hi there! How can I help?",
      "timestamp": "2024-06-30T10:00:02",
      "feedback": "like"
    }
  ]
}
```

### 4. Feedback System (`feedback_system.py`)

**Purpose**: Track feedback for continuous improvement

**Key Classes**:
- `Feedback`: Single feedback entry
- `FeedbackSession`: Feedback for one session
- `FeedbackManager`: Global feedback tracking

**Feedback Types**:
```
1. POSITIVE (Like) - Weight: 1.0
2. NEGATIVE (Dislike) - Weight: 1.0
3. TEACHER_INPUT - Weight: 2.0 (higher priority)
4. SNITCH_ALERT - Weight: 1.5
```

**Feedback Loop**:
```
User Interaction
    ↓
[Like/Dislike] → Feedback recorded
[Teacher Input] → Higher weight, direct guidance
[Snitch Alert] → Safety feedback
    ↓
Feedback Aggregated
    ↓
Feedback Ratio Calculated
    ↓
Training Adjusted (in next cycle)
```

## 🔄 Training Loop Detail

### Single Step in Environment

```python
# Per parallel env, per step:

1. Get observations from all agents
   obs = env.observe(all_agents)
   
2. For each agent:
   action, _ = model.predict(obs[agent_id])
   
3. Execute actions in physics:
   env.step(actions)
   
4. Calculate collisions:
   caught_hiders = check_collisions()
   
5. Reward computation:
   For hiders:
     if caught: reward = -1.0
     else: reward = 0.1
   For seekers:
     reward = 0.5 * num_found
   For snitch:
     if anomaly detected: reward -= 0.5
   
6. Detect reward hacking:
   suspicious = detect_reward_hacking()
   
7. Return:
   new_obs, rewards, dones, info
```

### PPO Update Cycle

```
Every 2048 steps (n_steps):

1. Compute advantages (GAE):
   advantage = reward + gamma * V(next_state) - V(state)
   
2. For 10 epochs (n_epochs):
   
   For each mini-batch (batch_size=64):
   
   a. Compute policy loss:
      ratio = new_policy / old_policy
      clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε=0.2
      policy_loss = -min(ratio*advantage, clipped_ratio*advantage)
      
   b. Compute value loss:
      value_loss = MSE(V(s), returns)
      
   c. Compute entropy loss:
      entropy = -Σ(p * log(p))
      
   d. Total loss:
      loss = policy_loss + 0.5*value_loss - 0.01*entropy
      
   e. Backprop and update
   
3. Save checkpoint every 100 episodes
```

## 🔐 Safety Layer (Snitch AI)

**Purpose**: Detect anomalous behaviors and reward hacking

**Detection Methods**:

1. **Boundary Violation**
   ```
   if |x| > map_size/2 or |y| > map_size/2:
     suspicious_score = 0.8
   ```

2. **Unnatural Velocity**
   ```
   if speed > max_speed * 1.5:
     suspicious_score = 0.6
   ```

3. **Repeated Actions**
   ```
   if same_action_repeated > threshold:
     suspicious_score = 0.4
   ```

4. **Reward Anomalies**
   ```
   if reward > expected_max * 2:
     suspicious_score = 0.7
   ```

**Action on Detection**:
```
if suspicious_score > threshold (0.7):
  1. Log alert
  2. Apply penalty: reward -= 0.5
  3. Flag in feedback system
  4. Continue monitoring
```

## 📊 Data Flow During Training

```
┌─────────────┐
│  Config     │
└──────┬──────┘
       ↓
┌─────────────────────────────┐
│ Environment × 4 (parallel)  │
│ - Game state                │
│ - Agent positions           │
│ - Collisions                │
└──────┬──────────────────────┘
       ↓
┌─────────────────────────────┐
│ PPO Agent × 3               │
│ - Student 1, Student 2      │
│ - Snitch Detector           │
└──────┬──────────────────────┘
       ↓
┌─────────────────────────────┐
│ Experience Buffer           │
│ - Observations              │
│ - Actions                   │
│ - Rewards                   │
│ - Done flags                │
└──────┬──────────────────────┘
       ↓ (every 2048 steps)
┌─────────────────────────────┐
│ Compute Returns & Advantages│
│ - Generalized Advantage Est.│
└──────┬──────────────────────┘
       ↓
┌─────────────────────────────┐
│ PPO Update × 10 epochs      │
│ - Mini-batch training       │
│ - Gradient clipping         │
│ - Loss computation          │
└──────┬──────────────────────┘
       ↓
┌─────────────────────────────┐
│ Model Checkpoint            │
│ - Save weights              │
│ - Log metrics               │
└─────────────────────────────┘
```

## 💾 Model Checkpoint Structure

Saved model (`.zip`) contains:
```
model.zip
├── model.pt          # PyTorch model weights
├── parameters.json   # Hyperparameters
├── policy_net.pkl    # Policy network
└── value_net.pkl     # Value network
```

## 🎯 Scaling Considerations

**Current (Tier 1)**:
- 4 parallel envs
- 3 agents (2 students + 1 snitch)
- 1M training steps
- Single GPU (RTX 3060/4070)

**Future (Tier 2)**:
- 100 parallel envs
- 8 student agents + 1 snitch
- 10M+ training steps
- Multi-GPU support
- Distributed training

**Performance Metrics**:
```
Tier 1 (Current):
  Training time: 2-4 hours
  VRAM usage: ~8GB
  Throughput: ~5000 steps/min
  
Tier 2 (Projected):
  Training time: 8-16 hours
  VRAM usage: ~24GB (3 GPUs)
  Throughput: ~20000 steps/min
```

## 🔌 Integration Points

**For Custom Training Data**:
```python
# In training.py, add:
def inject_teacher_knowledge(model, knowledge_data):
    # Teacher input integration point
    pass
```

**For Real-Time Monitoring**:
```python
# Subscribe to training events:
trainer.on_step.subscribe(lambda step: print(f"Step {step}"))
trainer.on_checkpoint.subscribe(lambda ckpt: save_to_cloud(ckpt))
```

**For Custom Environments**:
```python
# Implement Gym interface:
class CustomEnv(gym.Env):
    def __init__(self):
        # Must have action_space, observation_space
        pass
```

---

**This architecture supports scaling from Tier 1 (current) → Tier 2 → Tier 3 with minimal changes.**
