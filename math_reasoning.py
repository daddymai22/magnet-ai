"""
Math Reasoning Module for Agent Training

Teaches agents how to calculate and reason through mathematical problems
instead of just guessing. Integrates with PPO training for hybrid learning.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class MathProblem:
    """Represents a mathematical reasoning problem"""
    problem_id: str
    question: str
    operation: str  # 'addition', 'multiplication', 'geometry', etc.
    inputs: List[float]
    expected_output: float
    reasoning_steps: List[str]
    difficulty: int  # 1-10
    
    def to_dict(self) -> dict:
        return {
            'problem_id': self.problem_id,
            'question': self.question,
            'operation': self.operation,
            'inputs': self.inputs,
            'expected_output': self.expected_output,
            'reasoning_steps': self.reasoning_steps,
            'difficulty': self.difficulty
        }


class MathReasoningLayer(nn.Module):
    """
    Neural network layer that performs step-by-step mathematical reasoning.
    Mimics how humans solve math problems by breaking them into steps.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_steps: int = 5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        
        # Step processor: processes each reasoning step
        self.step_processor = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Step output generator: generates intermediate step results
        self.step_output = nn.Linear(hidden_size, 1)
        
        # Final calculator: combines all steps for final answer
        self.final_calculator = nn.Linear(hidden_size * num_steps, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that shows step-by-step reasoning.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            final_output: Final calculated answer
            step_outputs: List of intermediate step results
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        step_outputs = []
        
        # Process through multiple reasoning steps
        for step in range(self.num_steps):
            # Combine input with current hidden state
            combined = torch.cat([x, h], dim=1)
            
            # Process this step
            h = self.step_processor(combined)
            
            # Generate intermediate output for this step
            step_out = self.step_output(h)
            step_outputs.append(step_out)
        
        # Combine all step outputs for final answer
        all_steps = torch.cat(step_outputs, dim=1)
        final_output = self.final_calculator(all_steps)
        
        return final_output, step_outputs


class KnowledgeDistiller:
    """
    Transfers expert mathematical knowledge into agent policies.
    Uses teacher feedback to guide learning.
    """
    
    def __init__(self, reasoning_layer: MathReasoningLayer, learning_rate: float = 1e-4):
        self.reasoning_layer = reasoning_layer
        self.optimizer = torch.optim.Adam(
            reasoning_layer.parameters(), 
            lr=learning_rate
        )
        self.criterion = nn.MSELoss()
        self.training_history = []
        
    def teach_from_problem(self, problem: MathProblem) -> Dict:
        """
        Train the reasoning layer on a single mathematical problem.
        Mimics human teacher showing how to solve a problem.
        
        Args:
            problem: MathProblem instance with correct answer
            
        Returns:
            training_metrics: Loss and accuracy metrics
        """
        # Convert problem inputs to tensor
        inputs = torch.tensor(problem.inputs, dtype=torch.float32).unsqueeze(0)
        expected = torch.tensor([problem.expected_output], dtype=torch.float32)
        
        # Forward pass
        output, step_outputs = self.reasoning_layer(inputs)
        
        # Calculate loss
        loss = self.criterion(output.squeeze(), expected)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        error = abs(output.item() - problem.expected_output)
        accuracy = max(0.0, 1.0 - (error / max(abs(problem.expected_output), 1e-6)))
        
        metrics = {
            'problem_id': problem.problem_id,
            'loss': loss.item(),
            'accuracy': accuracy,
            'error': error,
            'expected': problem.expected_output,
            'predicted': output.item()
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def teach_batch(self, problems: List[MathProblem]) -> Dict:
        """
        Train on a batch of problems (more efficient).
        
        Args:
            problems: List of MathProblem instances
            
        Returns:
            batch_metrics: Aggregated metrics
        """
        total_loss = 0.0
        total_accuracy = 0.0
        
        for problem in problems:
            metrics = self.teach_from_problem(problem)
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
        
        batch_metrics = {
            'num_problems': len(problems),
            'avg_loss': total_loss / len(problems),
            'avg_accuracy': total_accuracy / len(problems)
        }
        
        return batch_metrics


class MathProblemGenerator:
    """
    Generates mathematical problems for agent training.
    Creates increasingly difficult problems for curriculum learning.
    """
    
    @staticmethod
    def generate_addition_problem(difficulty: int = 1) -> MathProblem:
        """Generate addition problem based on difficulty level"""
        max_val = 10 * difficulty
        a = np.random.randint(1, max_val)
        b = np.random.randint(1, max_val)
        
        return MathProblem(
            problem_id=f"add_{a}_{b}",
            question=f"What is {a} + {b}?",
            operation="addition",
            inputs=[float(a), float(b)],
            expected_output=float(a + b),
            reasoning_steps=[
                f"Identify operands: {a} and {b}",
                f"Apply addition operation",
                f"Result: {a + b}"
            ],
            difficulty=difficulty
        )
    
    @staticmethod
    def generate_multiplication_problem(difficulty: int = 1) -> MathProblem:
        """Generate multiplication problem"""
        max_val = 5 * difficulty
        a = np.random.randint(1, max_val)
        b = np.random.randint(1, max_val)
        
        return MathProblem(
            problem_id=f"mul_{a}_{b}",
            question=f"What is {a} × {b}?",
            operation="multiplication",
            inputs=[float(a), float(b)],
            expected_output=float(a * b),
            reasoning_steps=[
                f"Identify operands: {a} and {b}",
                f"Apply multiplication ({a} groups of {b})",
                f"Result: {a * b}"
            ],
            difficulty=difficulty
        )
    
    @staticmethod
    def generate_distance_problem(difficulty: int = 1) -> MathProblem:
        """Generate Euclidean distance calculation problem"""
        scale = 10 * difficulty
        x1 = np.random.uniform(-scale, scale)
        y1 = np.random.uniform(-scale, scale)
        x2 = np.random.uniform(-scale, scale)
        y2 = np.random.uniform(-scale, scale)
        
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return MathProblem(
            problem_id=f"dist_{x1:.1f}_{y1:.1f}_{x2:.1f}_{y2:.1f}",
            question=f"Distance between ({x1:.1f}, {y1:.1f}) and ({x2:.1f}, {y2:.1f})?",
            operation="geometry",
            inputs=[x1, y1, x2, y2],
            expected_output=distance,
            reasoning_steps=[
                f"Identify points: P1=({x1:.1f}, {y1:.1f}), P2=({x2:.1f}, {y2:.1f})",
                f"Calculate Δx = {x2:.1f} - {x1:.1f} = {x2-x1:.1f}",
                f"Calculate Δy = {y2:.1f} - {y1:.1f} = {y2-y1:.1f}",
                f"Apply distance formula: √(Δx² + Δy²)",
                f"Result: {distance:.2f}"
            ],
            difficulty=difficulty
        )
    
    @staticmethod
    def generate_curriculum(num_problems: int = 100) -> List[MathProblem]:
        """
        Generate curriculum of problems increasing in difficulty.
        Mimics how humans learn math: simple → complex.
        """
        problems = []
        problem_types = ['addition', 'multiplication', 'distance']
        
        for i in range(num_problems):
            difficulty = min(10, 1 + (i // 10))  # Increase difficulty every 10 problems
            problem_type = problem_types[i % len(problem_types)]
            
            if problem_type == 'addition':
                problem = MathProblemGenerator.generate_addition_problem(difficulty)
            elif problem_type == 'multiplication':
                problem = MathProblemGenerator.generate_multiplication_problem(difficulty)
            else:
                problem = MathProblemGenerator.generate_distance_problem(difficulty)
            
            problems.append(problem)
        
        return problems


class ReasoningAugmentedAgent(nn.Module):
    """
    Combines PPO policy network with mathematical reasoning layer.
    Agent uses reasoning for calculation tasks, RL for decision-making.
    """
    
    def __init__(self, state_size: int, action_size: int, reasoning_enabled: bool = True):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.reasoning_enabled = reasoning_enabled
        
        # Base policy network (for RL decisions)
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Value network (for advantage calculation)
        self.value_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Reasoning layer (for calculation tasks)
        if reasoning_enabled:
            self.reasoning_layer = MathReasoningLayer(state_size, hidden_size=128)
        else:
            self.reasoning_layer = None
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Policy network output (action logits)"""
        return self.policy_net(state)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Value network output for advantage calculation"""
        return self.value_net(state)
    
    def reason(self, problem_inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Use reasoning layer for mathematical calculation.
        Returns final answer and intermediate reasoning steps.
        """
        if not self.reasoning_enabled or self.reasoning_layer is None:
            raise RuntimeError("Reasoning layer not enabled")
        
        return self.reasoning_layer(problem_inputs)


def save_training_results(distiller: KnowledgeDistiller, output_path: str):
    """Save training history and metrics to file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'training_history': distiller.training_history,
            'num_trained': len(distiller.training_history)
        }, f, indent=2)


def load_reasoning_layer(model_path: str, device: str = 'cpu') -> MathReasoningLayer:
    """Load a saved reasoning layer model"""
    reasoning_layer = MathReasoningLayer(input_size=4, hidden_size=256)
    reasoning_layer.load_state_dict(torch.load(model_path, map_location=device))
    reasoning_layer.to(device)
    reasoning_layer.eval()
    return reasoning_layer
