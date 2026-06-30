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
│  │    Math Reasoning Layer (NEW!)                       │  │
│  │  - Step-by-step problem solving                     │  │
│  │  - Knowledge distillation                           │  │
│  │  - Curriculum learning                              │  │
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
│  │  - Reasoning Layer Weights                          │  │
│  │  - Conversation History                             │  │
│  │  - Training Logs                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Component Details

### 1. Training System (`training.py` and `training_with_reasoning.py`)

**Purpose**: Orchestrate multi-agent RL training, optionally with math reasoning

**Key Classes**:
- `MagnetAITrainer`: Original training coordinator (RL only)
- `MagnetAITrainerWithReasoning`: Enhanced trainer with math reasoning integration
- `MagnetTrainingCallback`: Progress tracking
- `ReasoningAugmentedCallback`: Tracks both RL and reasoning progress

**Workflow**:
```
Config Load → Env Creation → PPO Initialization → Training Loop
     ↓
  GPU Setup
     ↓
  Math Reasoning Initialization (if enabled)
     ↓
  Vectorized Environments (4 parallel)
     ↓
  Experience Collection
     ↓
  Policy Updates (every 2048 steps)
     ↓
  Interleaved Math Reasoning Training (optional)
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
  7. [OPTIONAL] Train reasoning layer on math problems
  8. Save checkpoint if needed
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

### 3. Math Reasoning Layer (`math_reasoning.py`) - NEW!

**Purpose**: Teach agents mathematical problem-solving through knowledge distillation

**Key Classes**:

#### `MathProblem`
Represents a mathematical problem with reasoning steps:
```python
@dataclass
class MathProblem:
    problem_id: str
    question: str
    operation: str  # 'addition', 'multiplication', 'geometry'
    inputs: List[float]
    expected_output: float
    reasoning_steps: List[str]
    difficulty: int  # 1-10
```

#### `MathReasoningLayer`
Neural network that performs step-by-step mathematical reasoning:
```python
class MathReasoningLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_steps: int = 5):
        # Step processor: processes each reasoning step
        self.step_processor = nn.Sequential(...)
        
        # Step output generator: generates intermediate results
        self.step_output = nn.Linear(...)
        
        # Final calculator: combines all steps
        self.final_calculator = nn.Linear(...)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Returns: final_output, intermediate_step_outputs
```

**Reasoning Architecture**:
```
Input Problem
    ↓
[Step 1: Extract features] → Output₁
    ↓
[Step 2: Intermediate calc] → Output₂
    ↓
[Step 3: Apply operation]  → Output₃
    ↓
[Step 4: Refine result]    → Output₄
    ↓
[Step 5: Combine logic]    → Output₅
    ↓
[Final Calculator: Aggregate] → Final Answer
```

#### `KnowledgeDistiller`
Transfers expert mathematical knowledge into agent policies:
```python
class KnowledgeDistiller:
    def teach_from_problem(self, problem: MathProblem) -> Dict:
        # Train reasoning layer on single problem
        
    def teach_batch(self, problems: List[MathProblem]) -> Dict:
        # Train on batch of problems (more efficient)
```

#### `MathProblemGenerator`
Generates curriculum of mathematical problems:
```python
class MathProblemGenerator:
    @staticmethod
    def generate_addition_problem(difficulty: int) -> MathProblem:
        # Addition problems with varying difficulty
        
    @staticmethod
    def generate_multiplication_problem(difficulty: int) -> MathProblem:
        # Multiplication problems
        
    @staticmethod
    def generate_distance_problem(difficulty: int) -> MathProblem:
        # Euclidean distance (useful for hide-and-seek)
        
    @staticmethod
    def generate_curriculum(num_problems: int) -> List[MathProblem]:
        # Curriculum: easy → complex
```

#### `ReasoningAugmentedAgent`
Combines PPO policy with reasoning capability:
```python
class ReasoningAugmentedAgent(nn.Module):
    def __init__(self, state_size: int, action_size: int, reasoning_enabled: bool):
        self.policy_net = ...  # RL decision-making
        self.value_net = ...   # Advantage calculation
        self.reasoning_layer = MathReasoningLayer(...)  # Math calculation
    
    def forward(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        # Policy + value outputs
        
    def reason(self, problem_inputs) -> Tuple[torch.Tensor, List]:
        # Mathematical reasoning with intermediate steps
```

**Problem Types**:

1. **Addition Problems**
   - Difficulty 1: 1-10 + 1-10
   - Difficulty 10: 1-100 + 1-100

2. **Multiplication Problems**
   - Difficulty 1: 1-5 × 1-5
   - Difficulty 10: 1-50 × 1-50

3. **Geometry Problems (Distance)**
   - Calculate Euclidean distance between two points
   - Useful for spatial reasoning in hide-and-seek
   - Shows intermediate calculation steps

**Curriculum Learning**:
```
Problems 0-9:   Difficulty 1 (easy)
Problems 10-19: Difficulty 2
Problems 20-29: Difficulty 3
...
Problems 90-99: Difficulty 10 (hard)
```

### 4. Chat Interface (`chat_interface.py`)

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

### 5. Feedback System (`feedback_system.py`)

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

### Math Reasoning Training Cycle (NEW!)

```
Every N RL updates (interleaved):

1. Sample batch of math problems from curriculum
   
2. For each problem in batch:
   
   a. Convert problem inputs to tensor
      inputs = problem.inputs
      expected = problem.expected_output
      
   b. Forward pass through reasoning layer
      output, step_outputs = reasoning_layer(inputs)
      
   c. Calculate loss
      loss = MSE(output, expected)
      
   d. Backward pass
      loss.backward()
      optimizer.step()
      
   e. Track accuracy
      accuracy = 1 - (error / max(expected, 1e-6))
      
3. Log metrics:
   - Average loss
   - Average accuracy
   - Problem difficulty distribution
   
4. Continue RL training with enhanced agent
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
       │ (every 500 PPO steps)
       ├─────→ ┌──────────────────────────┐
       │       │ Math Reasoning Update    │
       │       │ - Sample problem batch   │
       │       │ - Forward pass           │
       │       │ - Calculate loss         │
       │       │ - Backward pass          │
       │       └──────────────────────────┘
       ↓
┌─────────────────────────────┐
│ Model Checkpoint            │
│ - Save weights              │
│ - Save reasoning layer      │
│ - Log metrics               │
└─────────────────────────────┘
```

## 💾 Model Checkpoint Structure

### RL Model (`.zip`)
```
model.zip
├── model.pt          # PyTorch model weights
├── parameters.json   # Hyperparameters
├── policy_net.pkl    # Policy network
└── value_net.pkl     # Value network
```

### Reasoning Layer (`.pt`)
```
reasoning.pt           # Serialized reasoning layer state_dict
```

### Results (`.json`)
```json
{
  "training_history": [
    {
      "problem_id": "add_5_3",
      "loss": 0.0123,
      "accuracy": 0.98,
      "error": 0.08,
      "expected": 8.0,
      "predicted": 8.08
    }
  ],
  "num_trained": 100,
  "avg_accuracy": 0.95
}
```

## 🎯 Scaling Considerations

**Current (Tier 1)**:
- 4 parallel envs
- 3 agents (2 students + 1 snitch)
- 1M training steps
- 100 math problems
- Single GPU (RTX 3060/4070)

**Future (Tier 2)**:
- 100 parallel envs
- 8 student agents + 1 snitch
- 10M+ training steps
- 1000+ math problems
- Multi-GPU support
- Distributed training

**Future (Tier 3)**:
- 1000+ parallel envs
- 100+ student agents
- 100M+ training steps
- Custom problem generation
- Multi-node distributed training
- Real-time reasoning visualization

**Performance Metrics**:
```
Tier 1 (Current):
  RL Training time: 2-4 hours
  Math Training time: +1-2 hours
  Total time: 3-6 hours
  VRAM usage: ~8GB
  Throughput: ~5000 RL steps/min
  Reasoning accuracy: 95%+
  
Tier 2 (Projected):
  RL Training time: 8-16 hours
  Math Training time: +4-6 hours
  Total time: 12-22 hours
  VRAM usage: ~24GB (3 GPUs)
  Throughput: ~20000 RL steps/min
  Reasoning accuracy: 98%+
```

## 🔌 Integration Points

**For Custom Training Data**:
```python
# In training_with_reasoning.py, add:
custom_problems = [
    MathProblem(...),
    MathProblem(...),
]
trainer.math_problems.extend(custom_problems)
```

**For Real-Time Monitoring**:
```python
# Subscribe to training events:
trainer_callback.on_step.subscribe(lambda step: print(f"Step {step}"))
trainer_callback.on_checkpoint.subscribe(lambda ckpt: save_to_cloud(ckpt))
```

**For Custom Environments**:
```python
# Implement Gym interface:
class CustomEnv(gym.Env):
    def __init__(self):
        # Must have action_space, observation_space
        pass
```

**For Custom Reasoning Problems**:
```python
# Extend MathProblemGenerator:
class CustomProblemGenerator(MathProblemGenerator):
    @staticmethod
    def generate_custom_problem(difficulty: int) -> MathProblem:
        # Your custom problem logic
        return MathProblem(...)
```

---

**This architecture supports scaling from Tier 1 (current) → Tier 2 → Tier 3 with minimal changes.**

**Key Innovation**: Combining RL (for decision-making) with supervised learning (for reasoning) creates agents that can both make strategic decisions AND perform accurate calculations.
