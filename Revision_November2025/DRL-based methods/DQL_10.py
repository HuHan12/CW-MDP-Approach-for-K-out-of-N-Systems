"""
Distributed Q-Learning (DQL) for K-out-of-N Maintenance Optimization

This implementation follows the DQL approach where:
- Each component has its own independent agent
- Agents share the global reward signal but make decisions independently
- This avoids the curse of dimensionality (2^N action space)

Author: Based on Wang et al. (2014) and Zhou et al. (2022)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import time
import json


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class SystemConfig:
    """Configuration for the K-out-of-N system"""
    
    def __init__(self,
                 n_components=20,
                 n_levels=4,
                 K=6,
                 setup_cost=-2000.0,
                 maintenance_penalty=-100.0,
                 failure_penalty=-1200.0,
                 system_penalty=-2000.0,
                 normal_operation=0.0):
        
        self.n_components = n_components
        self.n_levels = n_levels
        self.K = K
        self.setup_cost = setup_cost
        self.maintenance_penalty = maintenance_penalty
        self.failure_penalty = failure_penalty
        self.system_penalty = system_penalty
        self.normal_operation = normal_operation
    
    def to_dict(self):
        return {
            'n_components': self.n_components,
            'n_levels': self.n_levels,
            'K': self.K,
            'setup_cost': self.setup_cost,
            'maintenance_penalty': self.maintenance_penalty,
            'failure_penalty': self.failure_penalty,
            'system_penalty': self.system_penalty,
            'normal_operation': self.normal_operation
        }
    
    def from_dict(d):
        """Create SystemConfig from dictionary"""
        return SystemConfig(
            n_components=d['n_components'],
            n_levels=d['n_levels'],
            K=d['K'],
            setup_cost=d['setup_cost'],
            maintenance_penalty=d['maintenance_penalty'],
            failure_penalty=d['failure_penalty'],
            system_penalty=d['system_penalty'],
            normal_operation=d['normal_operation']
        )


class DQLConfig:
    """Configuration for DQL training"""
    
    def __init__(self,
                 hidden_size=64,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.999,
                 buffer_capacity=50000,
                 batch_size=64,
                 target_update_freq=100,
                 use_double_dqn=True):
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
    
    def to_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'buffer_capacity': self.buffer_capacity,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'use_double_dqn': self.use_double_dqn
        }
    
    def from_dict(d):
        """Create DQLConfig from dictionary"""
        return DQLConfig(
            hidden_size=d['hidden_size'],
            learning_rate=d['learning_rate'],
            gamma=d['gamma'],
            epsilon_start=d['epsilon_start'],
            epsilon_end=d['epsilon_end'],
            epsilon_decay=d['epsilon_decay'],
            buffer_capacity=d['buffer_capacity'],
            batch_size=d['batch_size'],
            target_update_freq=d['target_update_freq'],
            use_double_dqn=d['use_double_dqn']
        )


# ============================================================================
# ENVIRONMENT
# ============================================================================

class KOutOfNEnvironment:
    """
    Multi-component K-out-of-N maintenance environment
    
    Supports:
    - Heterogeneous components (different transition matrices per component)
    - Configurable system size and failure threshold
    - Flexible cost structure
    """
    
    def __init__(self, config, transition_matrices=None):
        """
        Initialize environment
        
        Args:
            config: SystemConfig object with system parameters
            transition_matrices: List of transition matrices, one per component.
                                If None, uses default matrix for all components.
                                If single matrix, uses same for all components.
        """
        self.config = config
        self.n_components = config.n_components
        self.n_levels = config.n_levels
        self.K = config.K
        
        self.setup_cost = config.setup_cost
        self.maintenance_penalty = config.maintenance_penalty
        self.failure_penalty = config.failure_penalty
        self.system_penalty = config.system_penalty
        self.normal_operation = config.normal_operation
        
        # Set up transition matrices for each component
        self._setup_transition_matrices(transition_matrices)
        
        self.state = None
        self.reset()
    
    def _setup_transition_matrices(self, transition_matrices):
        """Setup transition matrices for all components"""
        
        # Default transition matrix
        default_matrix = np.array([
            [0.8571, 0.1429, 0.0,    0.0],
            [0.0,    0.8571, 0.1429, 0.0],
            [0.0,    0.0,    0.8,    0.2],
            [0.0,    0.0,    0.0,    1.0]
        ])
        
        if transition_matrices is None:
            # Use default for all components
            self.transition_matrices = [default_matrix.copy() for _ in range(self.n_components)]
        elif isinstance(transition_matrices, np.ndarray) and transition_matrices.ndim == 2:
            # Single matrix provided, use for all components
            self.transition_matrices = [transition_matrices.copy() for _ in range(self.n_components)]
        elif isinstance(transition_matrices, list):
            # List of matrices provided
            if len(transition_matrices) == 1:
                self.transition_matrices = [transition_matrices[0].copy() for _ in range(self.n_components)]
            elif len(transition_matrices) == self.n_components:
                self.transition_matrices = [m.copy() for m in transition_matrices]
            else:
                raise ValueError("Expected {} transition matrices, got {}".format(
                    self.n_components, len(transition_matrices)))
        else:
            raise ValueError("transition_matrices must be None, a single matrix, or a list of matrices")
        
        # Build 3D transition tensors for each component
        self._build_transition_tensors()
    
    def _build_transition_tensors(self):
        """Build 3D transition tensor for each component: T[s, s', a]"""
        self.T = []
        
        for i in range(self.n_components):
            T_i = np.zeros((self.n_levels, self.n_levels, 2))
            
            # Action 0 (no maintenance): use transition matrix
            T_i[:, :, 0] = self.transition_matrices[i].copy()
            # Failed state is absorbing under no maintenance
            T_i[self.n_levels - 1, :, 0] = 0.0
            T_i[self.n_levels - 1, self.n_levels - 1, 0] = 1.0
            
            # Action 1 (maintenance): return to state 1 and then transition
            for j in range(self.n_levels):
                T_i[j, :, 1] = self.transition_matrices[i][0, :]
            
            self.T.append(T_i)
    
    def reset(self):
        """Reset all components to state 1 (new)"""
        self.state = np.ones(self.n_components, dtype=np.int32)
        return self.state.copy()
    
    def reset_to_state(self, state):
        """Reset to a specific state"""
        self.state = state.copy()
        return self.state.copy()
    
    def step(self, action):
        """
        Execute action and return next state, reward, done, info
        
        Args:
            action: Array of shape (n_components,) with values in {0, 1}
                   0 = do nothing, 1 = maintain
        
        Returns:
            next_state: New system state
            reward: Global reward (negative cost)
            done: Always False (continuing task)
            info: Dictionary with additional information
        """
        current_state = self.state.copy()
        
        # Count failed components BEFORE action
        failed_components = np.sum(current_state == self.n_levels)
        
        # Calculate component-level rewards
        component_rewards = np.zeros(self.n_components, dtype=np.float32)
        
        repair_mask = (action == 1)
        failed_mask = (current_state == self.n_levels)
        
        # Corrective maintenance (repair failed component)
        corrective_mask = repair_mask & failed_mask
        component_rewards[corrective_mask] = self.failure_penalty + self.normal_operation
        
        # Preventive maintenance (repair non-failed component)
        preventive_mask = repair_mask & ~failed_mask
        component_rewards[preventive_mask] = self.maintenance_penalty + self.normal_operation
        
        # No repair
        no_repair_mask = ~repair_mask
        component_rewards[no_repair_mask] = self.normal_operation
        
        # System reward
        system_reward = np.sum(component_rewards)
        
        # Add setup cost if any maintenance
        if np.any(repair_mask):
            system_reward += self.setup_cost
        
        # Add system failure penalty
        if failed_components >= self.K:
            system_reward += self.system_penalty
        
        # Transition to next state
        next_state = np.zeros(self.n_components, dtype=np.int32)
        for i in range(self.n_components):
            s_i = current_state[i] - 1  # Convert to 0-indexed
            a_i = action[i]
            # Sample next state from transition distribution
            next_state[i] = np.random.choice(self.n_levels, p=self.T[i][s_i, :, a_i]) + 1
        
        self.state = next_state
        failed_after = np.sum(next_state == self.n_levels)
        
        info = {
            'failed_components_before': failed_components,
            'failed_components_after': failed_after,
            'maintenance_occurred': np.any(repair_mask),
            'n_maintained': np.sum(repair_mask),
            'system_failed': failed_components >= self.K,
            'component_rewards': component_rewards.copy(),
        }
        
        return next_state.copy(), float(system_reward), False, info
    
    def get_state(self):
        """Get current system state"""
        return self.state.copy()
    
    def print_info(self):
        """Print environment configuration"""
        print("\nEnvironment Configuration:")
        print("  N (components): {}".format(self.n_components))
        print("  Levels: {}".format(self.n_levels))
        print("  K (failure threshold): {}".format(self.K))
        print("  Setup cost: {}".format(self.setup_cost))
        print("  Maintenance penalty (C_m): {}".format(self.maintenance_penalty))
        print("  Failure penalty (C_f): {}".format(self.failure_penalty))
        print("  System penalty (C_b): {}".format(self.system_penalty))
        print("  Transition matrices: {} configured".format(len(self.transition_matrices)))


# ============================================================================
# LOCAL Q-NETWORK (Small network for single component)
# ============================================================================

class LocalQNetwork(nn.Module):
    """
    Q-Network for a single component
    
    Input: Component state (scalar, normalized)
    Output: Q-values for 2 actions [do nothing, maintain]
    """
    
    def __init__(self, n_levels=4, hidden_size=64, n_hidden_layers=2):
        super(LocalQNetwork, self).__init__()
        
        self.n_levels = n_levels
        self.n_actions = 2
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, self.n_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: Tensor of shape (batch_size, 1) or (batch_size,)
        
        Returns:
            Q-values of shape (batch_size, 2)
        """
        if state.dim() == 1:
            state = state.unsqueeze(1)
        return self.network(state)


# ============================================================================
# LOCAL REPLAY BUFFER (Per-component experience storage)
# ============================================================================

class LocalReplayBuffer:
    """
    Replay buffer for a single component
    Stores (s_i, a_i, R, s'_i) tuples where R is the GLOBAL reward
    """
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Store experience
        
        Args:
            state: Component state (1 to n_levels)
            action: Action taken (0 or 1)
            reward: GLOBAL reward from environment
            next_state: Next component state
            done: Episode done flag
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int64)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DQL AGENT (Single component agent)
# ============================================================================

class DQLAgent:
    """
    DQL Agent for a single component
    
    Each agent:
    - Maintains its own Q-network
    - Makes decisions based only on its own state
    - Updates using the global reward signal
    """
    
    def __init__(self, component_id, n_levels, config, device):
        
        self.component_id = component_id
        self.n_levels = n_levels
        self.config = config
        self.device = device
        
        # Networks
        self.q_network = LocalQNetwork(
            n_levels=n_levels,
            hidden_size=config.hidden_size
        ).to(device)
        
        self.target_network = LocalQNetwork(
            n_levels=n_levels,
            hidden_size=config.hidden_size
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = LocalReplayBuffer(capacity=config.buffer_capacity)
        
        # Exploration
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        
        # Training stats
        self.update_count = 0
        self.total_loss = 0.0
    
    def normalize_state(self, state):
        """Normalize state to [0, 1]"""
        return (state - 1) / (self.n_levels - 1)
    
    def select_action(self, state, mode='train'):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Component state (1 to n_levels)
            mode: 'train' for epsilon-greedy, 'eval' for greedy
        
        Returns:
            Action (0 or 1)
        """
        if mode == 'train' and random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            state_normalized = self.normalize_state(state)
            state_tensor = torch.FloatTensor([[state_normalized]]).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size):
        """
        Perform one gradient descent step
        
        Returns:
            Loss value or None if buffer too small
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Normalize states
        states_norm = (states - 1) / (self.n_levels - 1)
        next_states_norm = (next_states - 1) / (self.n_levels - 1)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states_norm).unsqueeze(1).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states_norm).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states_t).gather(1, actions_t).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use online network to select action, target to evaluate
                next_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states_t).gather(1, next_actions).squeeze()
            else:
                # Standard DQN
                next_q = self.target_network(next_states_t).max(dim=1)[0]
            
            target_q = rewards_t + (1 - dones_t) * self.config.gamma * next_q
        
        # Loss and optimization
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_count += 1
        self.total_loss += loss.item()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self, state):
        """Get Q-values for a state (for analysis)"""
        state_normalized = self.normalize_state(state)
        state_tensor = torch.FloatTensor([[state_normalized]]).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        return q_values
    
    def get_policy(self):
        """Get learned policy for all states"""
        policy = np.zeros(self.n_levels, dtype=np.int32)
        
        for s in range(1, self.n_levels + 1):
            q_values = self.get_q_values(s)
            policy[s - 1] = np.argmax(q_values)
        
        return policy


# ============================================================================
# DQL SYSTEM (Coordinates all agents)
# ============================================================================

class DQLSystem:
    """
    Distributed Q-Learning System
    
    Manages N independent agents, one per component.
    Key characteristics:
    - Each agent makes decisions independently based on local state
    - All agents receive the same global reward for learning
    - This captures the "cost sharing" aspect of DQL
    """
    
    def __init__(self, system_config, dql_config, device='auto'):
        
        self.system_config = system_config
        self.dql_config = dql_config
        self.n_components = system_config.n_components
        self.n_levels = system_config.n_levels
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print("DQL System using device: {}".format(self.device))
        
        # Create agents
        self.agents = []
        for i in range(self.n_components):
            agent = DQLAgent(
                component_id=i,
                n_levels=self.n_levels,
                config=dql_config,
                device=self.device
            )
            self.agents.append(agent)
        
        # Global epsilon (shared across agents for consistency)
        self.epsilon = dql_config.epsilon_start
        
        # Stats
        self.total_updates = 0
    
    def select_action(self, state, mode='train'):
        """
        Select actions for all components
        
        Args:
            state: System state array of shape (n_components,)
            mode: 'train' or 'eval'
        
        Returns:
            Action array of shape (n_components,)
        """
        actions = np.zeros(self.n_components, dtype=np.int32)
        
        for i, agent in enumerate(self.agents):
            # Override agent's epsilon with global epsilon for consistency
            if mode == 'train':
                agent.epsilon = self.epsilon
            actions[i] = agent.select_action(state[i], mode)
        
        return actions
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience for all agents
        
        CRITICAL: All agents receive the SAME global reward!
        This is the key feature of DQL.
        """
        for i, agent in enumerate(self.agents):
            agent.store_experience(
                state=state[i],
                action=action[i],
                reward=reward,  # GLOBAL reward for all agents
                next_state=next_state[i],
                done=done
            )
    
    def update_all_agents(self, batch_size):
        """
        Update all agents
        
        Returns:
            List of losses (one per agent)
        """
        losses = []
        
        for agent in self.agents:
            loss = agent.update(batch_size)
            losses.append(loss)
        
        self.total_updates += 1
        
        return losses
    
    def update_target_networks(self):
        """Update target networks for all agents"""
        for agent in self.agents:
            agent.update_target_network()
    
    def decay_epsilon(self):
        """Decay global exploration rate"""
        self.epsilon = max(
            self.dql_config.epsilon_end,
            self.epsilon * self.dql_config.epsilon_decay
        )
        # Sync to all agents
        for agent in self.agents:
            agent.epsilon = self.epsilon
    
    def get_policies(self):
        """
        Get learned policies for all components
        
        Returns:
            Array of shape (n_components, n_levels)
            policies[i, s-1] = action for component i in state s
        """
        policies = np.zeros((self.n_components, self.n_levels), dtype=np.int32)
        
        for i, agent in enumerate(self.agents):
            policies[i] = agent.get_policy()
        
        return policies
    
    def get_q_table(self):
        """
        Get Q-values for all components and states
        
        Returns:
            Array of shape (n_components, n_levels, 2)
            q_table[i, s-1, a] = Q_i(s, a)
        """
        q_table = np.zeros((self.n_components, self.n_levels, 2))
        
        for i, agent in enumerate(self.agents):
            for s in range(1, self.n_levels + 1):
                q_table[i, s - 1] = agent.get_q_values(s)
        
        return q_table
    
    def save(self, filepath):
        """Save all agents to file"""
        checkpoint = {
            'system_config': self.system_config.to_dict(),
            'dql_config': self.dql_config.to_dict(),
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'agents': []
        }
        
        for i, agent in enumerate(self.agents):
            agent_data = {
                'component_id': agent.component_id,
                'q_network': agent.q_network.state_dict(),
                'target_network': agent.target_network.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'update_count': agent.update_count
            }
            checkpoint['agents'].append(agent_data)
        
        torch.save(checkpoint, filepath)
        print("DQL System saved to {}".format(filepath))
    
    def load(self, filepath):
        """Load all agents from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.epsilon = checkpoint['epsilon']
        self.total_updates = checkpoint['total_updates']
        
        for i, agent_data in enumerate(checkpoint['agents']):
            self.agents[i].q_network.load_state_dict(agent_data['q_network'])
            self.agents[i].target_network.load_state_dict(agent_data['target_network'])
            self.agents[i].optimizer.load_state_dict(agent_data['optimizer'])
            self.agents[i].update_count = agent_data['update_count']
            self.agents[i].epsilon = self.epsilon
        
        print("DQL System loaded from {}".format(filepath))
    
    def print_info(self):
        """Print system information"""
        print("\nDQL System Configuration:")
        print("  Number of agents: {}".format(self.n_components))
        print("  States per component: {}".format(self.n_levels))
        print("  Actions per component: 2")
        print("  Total Q-table entries: {}".format(self.n_components * self.n_levels * 2))
        print("  Device: {}".format(self.device))
        
        # Network size
        total_params = sum(
            sum(p.numel() for p in agent.q_network.parameters())
            for agent in self.agents
        )
        print("  Total network parameters: {:,}".format(total_params))


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_dql(env, dql_system, n_episodes=5000, max_steps=100,
              print_freq=100, eval_freq=500, save_freq=1000,
              save_path='dql_checkpoint.pt'):
    """
    Train DQL system
    
    Args:
        env: K-out-of-N environment
        dql_system: DQL system with agents
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        print_freq: Print progress every N episodes
        eval_freq: Evaluate policy every N episodes
        save_freq: Save checkpoint every N episodes
        save_path: Path for saving checkpoints
    
    Returns:
        Dictionary with training history
    """
    config = dql_system.dql_config
    
    # Training history
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'epsilons': [],
        'eval_rewards': [],
        'eval_episodes': []
    }
    
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("STARTING DQL TRAINING")
    print("=" * 70)
    print("Episodes: {}".format(n_episodes))
    print("Steps per episode: {}".format(max_steps))
    print("Batch size: {}".format(config.batch_size))
    print("=" * 70 + "\n")
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Select action (each agent decides independently)
            action = dql_system.select_action(state, mode='train')
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store experience (all agents get the SAME global reward)
            dql_system.store_experience(state, action, reward, next_state, done)
            
            # Update all agents
            losses = dql_system.update_all_agents(config.batch_size)
            valid_losses = [l for l in losses if l is not None]
            if valid_losses:
                episode_losses.append(np.mean(valid_losses))
            
            # Accumulate reward
            episode_reward += reward * (config.gamma ** step)
            
            state = next_state
        
        # Decay epsilon
        dql_system.decay_epsilon()
        
        # Update target networks
        if (episode + 1) % config.target_update_freq == 0:
            dql_system.update_target_networks()
        
        # Record history
        history['episode_rewards'].append(episode_reward)
        history['epsilons'].append(dql_system.epsilon)
        if episode_losses:
            history['losses'].append(np.mean(episode_losses))
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(history['episode_rewards'][-100:])
            avg_loss = np.mean(history['losses'][-100:]) if history['losses'] else 0
            
            print("Ep {:5d}/{} | Reward: {:10.2f} | Loss: {:8.4f} | eps: {:.4f} | Time: {:.1f}min".format(
                episode + 1, n_episodes, avg_reward, avg_loss, dql_system.epsilon, elapsed / 60))
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_dql(env, dql_system, n_trials=100,
                                       max_steps=max_steps, verbose=False)
            history['eval_rewards'].append(eval_reward)
            history['eval_episodes'].append(episode + 1)
            print("         -> Evaluation reward: {:.2f}".format(eval_reward))
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = save_path.replace('.pt', '_ep{}.pt'.format(episode + 1))
            dql_system.save(checkpoint_path)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("Total time: {:.2f} minutes".format(total_time / 60))
    print("=" * 70 + "\n")
    
    # Save final model
    dql_system.save(save_path)
    
    return history


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_dql(env, dql_system, n_trials=1000, max_steps=100, verbose=True):
    """
    Evaluate learned DQL policy
    
    Returns:
        Mean discounted reward
    """
    gamma = dql_system.dql_config.gamma
    trial_rewards = []
    
    for trial in range(n_trials):
        state = env.reset()
        trial_reward = 0
        
        for step in range(max_steps):
            action = dql_system.select_action(state, mode='eval')
            next_state, reward, done, info = env.step(action)
            trial_reward += reward * (gamma ** step)
            state = next_state
        
        trial_rewards.append(trial_reward)
    
    mean_reward = np.mean(trial_rewards)
    std_reward = np.std(trial_rewards)
    
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS ({} trials, {} steps each)".format(n_trials, max_steps))
        print("=" * 70)
        print("Mean Reward:   {:10.2f} +/- {:.2f}".format(mean_reward, std_reward))
        print("Min Reward:    {:10.2f}".format(np.min(trial_rewards)))
        print("Max Reward:    {:10.2f}".format(np.max(trial_rewards)))
        print("Median:        {:10.2f}".format(np.median(trial_rewards)))
        print("=" * 70 + "\n")
    
    return mean_reward


# ============================================================================
# POLICY ANALYSIS
# ============================================================================

def analyze_policy(dql_system, verbose=True):
    """
    Analyze learned policies
    
    Returns:
        Dictionary with policy analysis
    """
    policies = dql_system.get_policies()
    q_table = dql_system.get_q_table()
    
    analysis = {
        'policies': policies,
        'q_table': q_table,
        'maintenance_thresholds': []
    }
    
    # Find maintenance threshold for each component
    # (first state where action = 1 is preferred)
    for i in range(dql_system.n_components):
        threshold = None
        for s in range(dql_system.n_levels):
            if policies[i, s] == 1:
                threshold = s + 1  # Convert to 1-indexed
                break
        analysis['maintenance_thresholds'].append(threshold)
    
    if verbose:
        print("\n" + "=" * 70)
        print("POLICY ANALYSIS")
        print("=" * 70)
        
        print("\nLearned Policies (0=do nothing, 1=maintain):")
        header = "{:<12}".format("Component")
        for s in range(1, dql_system.n_levels + 1):
            header += "State {:>2}  ".format(s)
        header += "Threshold"
        print(header)
        print("-" * 70)
        
        for i in range(dql_system.n_components):
            row = "Component {:<2}  ".format(i + 1)
            for s in range(dql_system.n_levels):
                action = policies[i, s]
                row += "   {}     ".format(action)
            threshold = analysis['maintenance_thresholds'][i]
            if threshold is not None:
                row += "   {}".format(threshold)
            else:
                row += "   Never"
            print(row)
        
        print("=" * 70 + "\n")
    
    return analysis


# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_results(history, system_config, save_path='dql_training.png'):
    """Plot training curves"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    ax1 = axes[0, 0]
    ax1.plot(history['episode_rewards'], alpha=0.3, label='Episode Reward', color='blue')
    window = 100
    if len(history['episode_rewards']) >= window:
        moving_avg = np.convolve(history['episode_rewards'],
                                  np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(history['episode_rewards'])), moving_avg,
                label='{}-Episode Moving Avg'.format(window), linewidth=2, color='red')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Discounted Reward')
    ax1.set_title('Training Rewards (N={}, K={})'.format(
        system_config.n_components, system_config.K))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Losses
    ax2 = axes[0, 1]
    if history['losses']:
        ax2.plot(history['losses'], alpha=0.3, label='Loss', color='orange')
        if len(history['losses']) >= window:
            moving_avg_loss = np.convolve(history['losses'],
                                          np.ones(window) / window, mode='valid')
            ax2.plot(range(window - 1, len(history['losses'])), moving_avg_loss,
                    label='{}-Episode Moving Avg'.format(window), linewidth=2, color='green')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Epsilon decay
    ax3 = axes[1, 0]
    ax3.plot(history['epsilons'], color='purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate Decay')
    ax3.grid(True, alpha=0.3)
    
    # Evaluation rewards
    ax4 = axes[1, 1]
    if history['eval_rewards']:
        ax4.plot(history['eval_episodes'], history['eval_rewards'],
                'o-', color='green', linewidth=2, markersize=6)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Mean Evaluation Reward')
        ax4.set_title('Evaluation Performance')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print("Training plot saved to {}".format(save_path))
    plt.close()


def plot_policy_heatmap(dql_system, save_path='dql_policy.png'):
    """Plot policy as heatmap"""
    
    policies = dql_system.get_policies()
    
    fig, ax = plt.subplots(figsize=(12, max(6, dql_system.n_components * 0.3)))
    
    im = ax.imshow(policies, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Component', fontsize=12)
    ax.set_title('Learned Maintenance Policy\n(Green=Do Nothing, Red=Maintain)', fontsize=14)
    
    ax.set_xticks(range(dql_system.n_levels))
    ax.set_xticklabels(['{}'.format(s + 1) for s in range(dql_system.n_levels)])
    
    if dql_system.n_components <= 30:
        ax.set_yticks(range(dql_system.n_components))
        ax.set_yticklabels(['{}'.format(i + 1) for i in range(dql_system.n_components)])
    
    plt.colorbar(im, ax=ax, label='Action (0=None, 1=Maintain)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print("Policy heatmap saved to {}".format(save_path))
    plt.close()


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(system_config, dql_config, transition_matrices=None,
                   n_episodes=5000, max_steps=100, n_eval_trials=1000,
                   experiment_name='dql_experiment', seed=42):
    """
    Run a complete DQL experiment
    
    Args:
        system_config: System configuration
        dql_config: DQL training configuration
        transition_matrices: Transition matrices for components
        n_episodes: Training episodes
        max_steps: Steps per episode
        n_eval_trials: Evaluation trials
        experiment_name: Name for saving results
        seed: Random seed
    
    Returns:
        Dictionary with all results
    """
    # Set seed
    set_seed(seed)
    
    print("\n" + "=" * 70)
    print("DQL EXPERIMENT: {}".format(experiment_name))
    print("=" * 70)
    
    # Create environment
    env = KOutOfNEnvironment(
        config=system_config,
        transition_matrices=transition_matrices
    )
    env.print_info()
    
    # Create DQL system
    dql_system = DQLSystem(
        system_config=system_config,
        dql_config=dql_config
    )
    dql_system.print_info()
    
    # Training
    history = train_dql(
        env=env,
        dql_system=dql_system,
        n_episodes=n_episodes,
        max_steps=max_steps,
        print_freq=100,
        eval_freq=500,
        save_freq=n_episodes,  # Save only at end
        save_path='{}.pt'.format(experiment_name)
    )
    
    # Final evaluation
    eval_reward = evaluate_dql(
        env=env,
        dql_system=dql_system,
        n_trials=n_eval_trials,
        max_steps=max_steps,
        verbose=True
    )
    
    # Policy analysis
    analysis = analyze_policy(dql_system, verbose=True)
    
    # Plotting
    plot_training_results(history, system_config, '{}_training.png'.format(experiment_name))
    plot_policy_heatmap(dql_system, '{}_policy.png'.format(experiment_name))
    
    # Compile results
    results = {
        'experiment_name': experiment_name,
        'system_config': system_config.to_dict(),
        'dql_config': dql_config.to_dict(),
        'history': history,
        'final_eval_reward': eval_reward,
        'policies': analysis['policies'].tolist(),
        'maintenance_thresholds': analysis['maintenance_thresholds'],
        'seed': seed
    }
    
    # Save results
    results_path = '{}_results.json'.format(experiment_name)
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'experiment_name': results['experiment_name'],
            'system_config': results['system_config'],
            'dql_config': results['dql_config'],
            'final_eval_reward': float(results['final_eval_reward']),
            'policies': results['policies'],
            'maintenance_thresholds': results['maintenance_thresholds'],
            'seed': results['seed']
        }
        json.dump(serializable_results, f, indent=2)
    print("Results saved to {}".format(results_path))
    
    return results, env, dql_system


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function demonstrating DQL for K-out-of-N maintenance
    """
    print("\n" + "=" * 70)
    print("DISTRIBUTED Q-LEARNING (DQL)")
    print("For K-out-of-N Maintenance Optimization")
    print("=" * 70)
    
    # System configuration
    system_config = SystemConfig(
        n_components=10,
        n_levels=4,
        K=4,
        setup_cost=-2000.0,
        maintenance_penalty=-100.0,
        failure_penalty=-1200.0,
        system_penalty=-2000.0,
        normal_operation=0.0
    )
    
    # DQL configuration
    dql_config = DQLConfig(
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999,
        buffer_capacity=50000,
        batch_size=64,
        target_update_freq=100,
        use_double_dqn=True
    )
    
    # Default transition matrix
    transition_matrix = np.array([
        [0.8571, 0.1429, 0.0,    0.0],
        [0.0,    0.8571, 0.1429, 0.0],
        [0.0,    0.0,    0.8,    0.2],
        [0.0,    0.0,    0.0,    1.0]
    ])
    
    # Run experiment
    results, env, dql_system = run_experiment(
        system_config=system_config,
        dql_config=dql_config,
        transition_matrices=transition_matrix,
        n_episodes=10000,
        max_steps=100,
        n_eval_trials=1000,
        experiment_name='dql_N10_K4',
        seed=42
    )
    
    print("\nFinal Evaluation Reward: {:.2f}".format(results['final_eval_reward']))
    
    return results, env, dql_system


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # System check
    print("System Check:")
    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
    print()
    
    # Run main experiment
    results, env, dql_system = main()