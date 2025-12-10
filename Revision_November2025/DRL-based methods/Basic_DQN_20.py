import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List
import matplotlib.pyplot as plt
import time
import gc


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA optimizations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)


# ============================================================================
# BLOCK 1: ENVIRONMENT (Same as before)
# ============================================================================

class KOutOfNEnvironment:
    """
    Multi-component K-out-of-N maintenance environment
    Optimized for N=20, K=6
    """
    def __init__(self, 
                 n_components: int = 20,
                 n_levels: int = 4,
                 K: int = 6,
                 transition_matrix: np.ndarray = None,
                 setup_cost: float = -2000.0,
                 maintenance_penalty: float = -100.0,
                 failure_penalty: float = -1200.0,
                 system_penalty: float = -2000.0,
                 normal_operation: float = 0.0):
        
        self.n_components = n_components
        self.n_levels = n_levels
        self.K = K
       
        self.setup_cost = setup_cost
        self.maintenance_penalty = maintenance_penalty
        self.failure_penalty = failure_penalty
        self.system_penalty = system_penalty
        self.normal_operation = normal_operation

        if transition_matrix is None:
            self.transition_matrix = np.array([
                [0.8571, 0.1429, 0.0,    0.0],
                [0.0,    0.8571, 0.1429, 0.0],
                [0.0,    0.0,    0.8,    0.2],
                [0.0,    0.0,    0.0,    1.0]
            ])
        else:
            self.transition_matrix = transition_matrix

        self._build_transition_tensor()
        self.state = None
        self.reset()

    def _build_transition_tensor(self):
        """Build 3D transition tensor"""
        self.T = np.zeros((self.n_levels, self.n_levels, 2))
        
        # Action 0 (no repair)
        self.T[:, :, 0] = self.transition_matrix.copy()
        self.T[self.n_levels-1, :, 0] = 0.0
        self.T[self.n_levels-1, self.n_levels-1, 0] = 1.0
        
        # Action 1 (repair)
        for j in range(self.n_levels):
            self.T[j, :, 1] = self.transition_matrix[0, :]

    def reset(self) -> np.ndarray:
        """Reset all components to state 1"""
        self.state = np.ones(self.n_components, dtype=np.int32)
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return next state, reward, done, info"""
        current_state = self.state.copy()
        
        # Count failed components BEFORE transition
        failed_components = np.sum(current_state == self.n_levels)
        
        # Calculate component-level rewards (vectorized for speed)
        component_rewards = np.zeros(self.n_components, dtype=np.float32)
        
        # Repair actions
        repair_mask = (action == 1)
        failed_mask = (current_state == self.n_levels)
        
        # Corrective maintenance
        corrective_mask = repair_mask & failed_mask
        component_rewards[corrective_mask] = self.failure_penalty + self.normal_operation
        
        # Preventive maintenance
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
        
        # Transition to next state (vectorized)
        next_state = np.zeros(self.n_components, dtype=np.int32)
        for i in range(self.n_components):
            s_i = current_state[i] - 1
            a_i = action[i]
            next_state[i] = np.random.choice(self.n_levels, p=self.T[s_i, :, a_i]) + 1
        
        self.state = next_state
        failed_after = np.sum(next_state == self.n_levels)
        
        info = {
            'failed_components_before': failed_components,
            'failed_components_after': failed_after,
            'maintenance_occurred': np.any(repair_mask),
            'system_failed': failed_components >= self.K,
        }
        
        return next_state.copy(), float(system_reward), False, info
    
    def get_state(self) -> np.ndarray:
        return self.state.copy()
    
    def print_info(self):
        print(f"Environment Configuration:")
        print(f"  N (components): {self.n_components}")
        print(f"  Levels: {self.n_levels}")
        print(f"  K (failure threshold): {self.K}")
        print(f"  Setup cost: {self.setup_cost}")
        print(f"  Maintenance penalty: {self.maintenance_penalty}")
        print(f"  Failure penalty: {self.failure_penalty}")
        print(f"  System penalty: {self.system_penalty}")


# ============================================================================
# BLOCK 2: STATE/ACTION ENCODER (Optimized for large N)
# ============================================================================

class StateActionEncoder:
    """
    Optimized encoder for N=20
    WARNING: Action space is 2^20 = 1,048,576 actions!
    """
    
    def __init__(self, n_components: int = 20, n_levels: int = 4):
        self.n_components = n_components
        self.n_levels = n_levels
        self.n_states = n_levels ** n_components  # Will be huge!
        self.n_actions = 2 ** n_components  # 1,048,576 for N=20
        
        print(f"\n⚠️  WARNING: Action space size = {self.n_actions:,}")
        print(f"⚠️  WARNING: This is extremely large for vanilla DQN!")
        
        # DON'T pre-generate all actions (too many!)
        self._all_actions = None
    
    def state_to_index(self, state: np.ndarray) -> int:
        """Convert state to index (base-4 encoding)"""
        state_0indexed = state - 1
        index = 0
        for i, s in enumerate(state_0indexed):
            index += int(s) * (self.n_levels ** i)
        return int(index)
    
    def index_to_state(self, index: int) -> np.ndarray:
        """Convert index to state"""
        state = np.zeros(self.n_components, dtype=np.int32)
        remaining = index
        for i in range(self.n_components):
            state[i] = remaining % self.n_levels
            remaining //= self.n_levels
        return state + 1
    
    def action_to_index(self, action: np.ndarray) -> int:
        """Convert action to index (binary encoding)"""
        index = 0
        for i, a in enumerate(action):
            if a == 1:
                index += (1 << i)  # Bit shifting (faster than 2**i)
        return int(index)
    
    def index_to_action(self, index: int) -> np.ndarray:
        """Convert index to action (optimized with bit operations)"""
        action = np.zeros(self.n_components, dtype=np.int32)
        for i in range(self.n_components):
            action[i] = (index >> i) & 1  # Bit extraction
        return action
    
    def print_info(self):
        print(f"Encoder Configuration:")
        print(f"  State space size: {self.n_states} (4^{self.n_components})")
        print(f"  Action space size: {self.n_actions:,} (2^{self.n_components})")


# ============================================================================
# BLOCK 3: Q-NETWORK (Optimized for GPU and large output)
# ============================================================================


class QNetwork(nn.Module):
    """
    Q-Network optimized for N=20 (1M actions output)
    """
    def __init__(self, 
                 n_components: int = 20, 
                 n_levels: int = 4, 
                 n_actions: int = 1048576,
                 hidden_size: int = 512):
        super(QNetwork, self).__init__()
        
        self.n_components = n_components
        self.n_levels = n_levels
        self.n_actions = n_actions
        
        # Network WITHOUT BatchNorm
        self.network = nn.Sequential(
            nn.Linear(n_components, hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
           
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
          
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with smaller weights for large output layer"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

# class QNetwork(nn.Module):
#     """
#     Q-Network optimized for N=20 (1M actions output)
#     Uses larger hidden layers to handle complexity
#     """
#     def __init__(self, 
#                  n_components: int = 20, 
#                  n_levels: int = 4, 
#                  n_actions: int = 1048576,
#                  hidden_size: int = 512):  # Larger for N=20
#         super(QNetwork, self).__init__()
        
#         self.n_components = n_components
#         self.n_levels = n_levels
#         self.n_actions = n_actions
        
#         # Deeper network for complex problem
#         self.network = nn.Sequential(
#             nn.Linear(n_components, hidden_size),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_size),  # Batch norm for stability
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_size),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_size),
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, n_actions)  # HUGE output layer!
#         )
        
#         # Careful initialization for large output
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         """Initialize with smaller weights for large output layer"""
#         for layer in self.network:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Smaller gain
#                 nn.init.constant_(layer.bias, 0.0)
    
#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass
#         Input: (batch_size, n_components)
#         Output: (batch_size, n_actions)  <- 1M actions for N=20!
#         """
#         return self.network(state)


# ============================================================================
# BLOCK 4: REPLAY BUFFER (Same as before)
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity: int = 100000):  # Larger buffer
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action_idx: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action_idx, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
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
# BLOCK 5: DQN AGENT (GPU-optimized)
# ============================================================================

class DQNAgent:
    """
    DQN Agent optimized for GPU and large action space
    """
    def __init__(self, 
                 n_components: int = 20,
                 n_levels: int = 4,
                 n_actions: int = 1048576,
                 hidden_size: int = 512,
                 learning_rate: float = 0.0001,  # Lower LR for stability
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.9995,  # Slower decay for large action space
                 buffer_capacity: int = 100000,
                 device: str = 'cuda'):
        
        self.n_components = n_components
        self.n_levels = n_levels
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Force CUDA if available
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        print(f"✓ Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Networks
        print("Building Q-network...")
        self.q_network = QNetwork(n_components, n_levels, n_actions, hidden_size).to(self.device)
        self.target_network = QNetwork(n_components, n_levels, n_actions, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Calculate model size
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"✓ Q-Network parameters: {total_params:,}")
        print(f"✓ Estimated model size: {total_params * 4 / 1e9:.2f} GB (float32)")
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training stats
        self.update_count = 0
    
    def normalize_state(self, state: np.ndarray) -> torch.Tensor:
        """Normalize state to [0, 1]"""
        normalized = (state - 1) / (self.n_levels - 1)
        return torch.FloatTensor(normalized).to(self.device)
    
    def select_action(self, state: np.ndarray, mode: str = 'train') -> int:
        """
        Epsilon-greedy action selection
        WARNING: With 1M actions, random exploration is nearly useless!
        """
        if mode == 'train' and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            state_tensor = self.normalize_state(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)  # (1, 1M) tensor!
                action_idx = q_values.argmax(dim=1).item()
            return action_idx
    
    def update(self, batch_size: int = 128):  # Larger batch for GPU
        """Gradient descent step (optimized for GPU)"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to GPU tensors
        states = torch.FloatTensor((states - 1) / (self.n_levels - 1)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor((next_states - 1) / (self.n_levels - 1)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]  # Max over 1M actions!
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Huber loss (more stable than MSE for large values)
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_count += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        print(f"✓ Model loaded from {filepath}")


# ============================================================================
# BLOCK 6: TRAINING LOOP (GPU-optimized with progress tracking)
# ============================================================================

def train_dqn(env: KOutOfNEnvironment,
              agent: DQNAgent,
              encoder: StateActionEncoder,
              n_episodes: int = 10000,
              max_steps: int = 100,
              batch_size: int = 128,
              target_update_freq: int = 50,  # Less frequent for stability
              print_freq: int = 50,
              save_freq: int = 1000):
    """
    GPU-optimized training loop with progress tracking
    """
    episode_rewards = []
    losses = []
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Episodes: {n_episodes}")
    print(f"Steps per episode: {max_steps}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*70}\n")
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action
            action_idx = agent.select_action(state, mode='train')
            action = encoder.index_to_action(action_idx)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store in replay buffer
            agent.replay_buffer.push(state, action_idx, reward, next_state, done)
            
            # Update Q-network
            loss = agent.update(batch_size)
            if loss is not None:
                episode_loss.append(loss)
            
            # Accumulate discounted reward
            episode_reward += reward * (agent.gamma ** step)
            
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        if len(episode_loss) > 0:
            losses.append(np.mean(episode_loss))
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses) if losses else 0
            
            print(f"Ep {episode+1:5d}/{n_episodes} | "
                  f"Reward: {avg_reward:8.2f} | "
                  f"Loss: {avg_loss:10.2f} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Buffer: {len(agent.replay_buffer):6d} | "
                  f"Time: {elapsed/60:.1f}min")
            
            # GPU memory stats
            if agent.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"         GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(f'dqn_checkpoint_ep{episode+1}.pt')
        
        # Memory cleanup (prevent memory leaks on GPU)
        if (episode + 1) % 100 == 0 and agent.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"{'='*70}\n")
    
    return episode_rewards, losses


# ============================================================================
# BLOCK 7: EVALUATION
# ============================================================================

def evaluate_policy(env: KOutOfNEnvironment,
                   agent: DQNAgent,
                   encoder: StateActionEncoder,
                   n_trials: int = 1000,
                   max_steps: int = 100,
                   verbose: bool = True):
    """Evaluate learned policy"""
    trial_rewards = []
    
    if verbose:
        print("Evaluating policy...")
    
    for trial in range(n_trials):
        state = env.reset()
        trial_reward = 0
        
        for step in range(max_steps):
            action_idx = agent.select_action(state, mode='eval')
            action = encoder.index_to_action(action_idx)
            next_state, reward, done, info = env.step(action)
            trial_reward += reward * (agent.gamma ** step)
            state = next_state
        
        trial_rewards.append(trial_reward)
        
        if verbose and (trial + 1) % 100 == 0:
            print(f"  Trial {trial+1}/{n_trials} complete...")
    
    mean_reward = np.mean(trial_rewards)
    std_reward = np.std(trial_rewards)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS ({n_trials} trials, {max_steps} steps each)")
        print(f"{'='*70}")
        print(f"Mean Reward: {mean_reward:10.2f} ± {std_reward:.2f}")
        print(f"Min Reward:  {np.min(trial_rewards):10.2f}")
        print(f"Max Reward:  {np.max(trial_rewards):10.2f}")
        print(f"Median:      {np.median(trial_rewards):10.2f}")
        print(f"{'='*70}\n")
    
    return trial_rewards, mean_reward, std_reward


# ============================================================================
# BLOCK 8: PLOTTING
# ============================================================================

def plot_training_results(episode_rewards, losses, n_components, K):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                label=f'{window}-Episode Moving Average', linewidth=2, color='red')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Discounted Reward', fontsize=12)
    ax1.set_title(f'Training Rewards (N={n_components}, K={K})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    if len(losses) > 0:
        ax2.plot(losses, alpha=0.3, label='Loss', color='orange')
        if len(losses) >= window:
            moving_avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(losses)), moving_avg_loss,
                    label=f'{window}-Episode Moving Average', linewidth=2, color='green')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'dqn_training_N{n_components}_K{K}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved as '{filename}'")
    plt.close()


# ============================================================================
# BLOCK 9: MAIN FUNCTION FOR N=20, K=6
# ============================================================================

def main():
    """
    Main function for N=20 components, K=6 failure threshold
    """
    print("\n" + "="*70)
    print("DQN FOR K-OUT-OF-N MAINTENANCE PROBLEM")
    print("="*70)
    print("Configuration: N=20 components, K=6 failure threshold")
    print("="*70 + "\n")
    
    # Problem parameters
    n_components = 20
    n_levels = 4
    K = 6
    
    # Cost structure (as specified)
    setup_cost = -2000.0
    maintenance_penalty = -100.0
    failure_penalty = -1200.0
    system_penalty = -2000.0
    normal_operation = 0.0
    
    # Create environment
    print("Setting up environment...")
    env = KOutOfNEnvironment(
        n_components=n_components,
        n_levels=n_levels,
        K=K,
        setup_cost=setup_cost,
        maintenance_penalty=maintenance_penalty,
        failure_penalty=failure_penalty,
        system_penalty=system_penalty,
        normal_operation=normal_operation
    )
    env.print_info()
    
    # Create encoder
    print("\nSetting up encoder...")
    encoder = StateActionEncoder(n_components=n_components, n_levels=n_levels)
    encoder.print_info()
    
    # Create DQN agent
    print("\nSetting up DQN agent...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    agent = DQNAgent(
        n_components=n_components,
        n_levels=n_levels,
        n_actions=encoder.n_actions,
        hidden_size=512,  # Large for complex problem
        learning_rate=0.0001,  # Lower LR for stability
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,  # Slower decay
        buffer_capacity=100000,  # Large buffer
        device=device
    )
    
    # Training
    print("\n" + "="*70)
    print("READY TO TRAIN")
    print("="*70)
    print("⚠️  WARNING: This will take a LONG time!")
    print("⚠️  With 1M actions, vanilla DQN will struggle significantly.")
    print("⚠️  Consider using Ctrl+C to stop early if no progress.")
    print("="*70 + "\n")
    
    input("Press Enter to start training (or Ctrl+C to abort)...")
    
    try:
        episode_rewards, losses = train_dqn(
            env=env,
            agent=agent,
            encoder=encoder,
            n_episodes=10000,  # May need more for convergence
            max_steps=100,
            batch_size=128,  # Larger batch for GPU
            target_update_freq=50,
            print_freq=50,
            save_freq=1000
        )
        
        # Save final model
        agent.save('dqn_final_N20_K6.pt')
        
        # Evaluation
        print("\nEvaluating final policy...")
        trial_rewards, mean_reward, std_reward = evaluate_policy(
            env=env,
            agent=agent,
            encoder=encoder,
            n_trials=1000,
            max_steps=100,
            verbose=True
        )
        
        # Plot results
        plot_training_results(episode_rewards, losses, n_components, K)
        
        return env, agent, encoder, episode_rewards, trial_rewards
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current model...")
        agent.save('dqn_interrupted_N20_K6.pt')
        return env, agent, encoder, None, None


# ============================================================================
# BLOCK 10: RUN
# ============================================================================

if __name__ == "__main__":
    # Check CUDA availability
    print("System Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run main
    env, agent, encoder, episode_rewards, trial_rewards = main()