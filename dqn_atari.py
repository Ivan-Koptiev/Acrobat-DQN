import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== DQN Acrobot (Slightly Larger Net) ===\n")

# --- DQN Model for Acrobot ---
class DQN(nn.Module):
    """
    Feedforward DQN for Acrobot environment (input: 6 state dimensions)
    """
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # For Acrobot: input_shape is 6 (state dimensions), n_actions is 3
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Added one more layer for better capacity
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.fc(x)

# --- DQN Agent ---
class DQNAgent:
    """
    DQN Agent implementing the core reinforcement learning algorithm.
    Includes experience replay, target network, and epsilon-greedy exploration.
    """
    def __init__(self, state_shape, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        # Improved exploration parameters for better final performance
        self.epsilon = 1.0  # Start with 100% exploration
        self.epsilon_min = 0.02  # Slightly higher minimum for better exploration
        self.epsilon_decay = 0.998  # Slower decay for more exploration
        
        # Learning parameters
        self.gamma = 0.99  # Discount factor for future rewards
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Neural networks
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Training statistics
        self.training_losses = []
        self.episode_rewards = []
        self.epsilon_history = []
        
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        With probability epsilon: random action (exploration)
        With probability 1-epsilon: best action according to Q-values (exploitation)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Train the network using experience replay.
        Sample random batch from memory and update Q-values.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values for the actions taken
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (using target network for stability)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        
        # Target Q-values: reward + gamma * max(Q(next_state)) if not done
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update network
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Store loss for visualization
        self.training_losses.append(loss.item())
        
        # Decay epsilon (reduce exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- Training Loop for Acrobot ---
def train_dqn(env_name="Acrobot-v1", n_episodes=800, render_every=100):
    """
    Train DQN agent on specified environment.
    
    Args:
        env_name: Gym environment name
        n_episodes: Number of training episodes
        render_every: Render environment every N episodes (for visualization)
    """
    print(f"Training DQN on {env_name}")
    print(f"Episodes: {n_episodes}")
    
    # Create environment
    env = gym.make(env_name)
    state_shape = env.observation_space.shape[0]  # For Acrobot: 6
    n_actions = env.action_space.n  # For Acrobot: 3
    
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Setup device and agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = DQNAgent(state_shape, n_actions, device)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    avg_rewards = []
    best_avg_reward = float('-inf')
    
    print("\n=== Starting Training ===\n")
    
    for episode in range(n_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # New gym API
        else:
            state = reset_result  # Old gym API
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            step_result = env.step(action)
            if len(step_result) == 5:  # New gym API
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:  # Old gym API
                next_state, reward, done, _ = step_result
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.replay()
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        agent.episode_rewards.append(episode_reward)
        agent.epsilon_history.append(agent.epsilon)
        
        # Calculate moving average
        if episode >= 99:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)
            
            # Track best performance
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        else:
            avg_rewards.append(np.mean(episode_rewards))
        
        # Print progress
        if episode % 50 == 0:
            print(f"Episode: {episode:3d}, Reward: {episode_reward:6.2f}, "
                  f"Length: {episode_length:3d}, Epsilon: {agent.epsilon:.3f}, "
                  f"Avg Reward: {avg_rewards[-1]:.2f}")
        
        # Early stopping if solved
        if episode >= 200 and avg_rewards[-1] >= -100:
            print(f"\nEnvironment solved at episode {episode}!")
            break
    
    env.close()
    
    # Save training statistics
    training_stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_rewards': avg_rewards,
        'epsilon_history': agent.epsilon_history,
        'training_losses': agent.training_losses,
        'final_epsilon': agent.epsilon,
        'best_avg_reward': best_avg_reward,
        'final_avg_reward': avg_rewards[-1]
    }
    
    return agent, training_stats

def evaluate_agent(agent, env_name="Acrobot-v1", n_episodes=20):
    """
    Evaluate trained agent without exploration (epsilon = 0).
    Increased episodes for better evaluation.
    """
    print(f"\n=== Evaluating Agent ===\n")
    
    env = gym.make(env_name)
    evaluation_rewards = []
    
    for episode in range(n_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # New gym API
        else:
            state = reset_result  # Old gym API
        episode_reward = 0
        done = False
        
        while not done:
            # Use greedy policy (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
            
            step_result = env.step(action)
            if len(step_result) == 5:  # New gym API
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:  # Old gym API
                state, reward, done, _ = step_result
            episode_reward += reward
        
        evaluation_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    avg_eval_reward = np.mean(evaluation_rewards)
    print(f"\nAverage evaluation reward: {avg_eval_reward:.2f}")
    
    return evaluation_rewards

def create_visualizations(training_stats, evaluation_rewards):
    print("\n=== Creating Visualizations ===\n")
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    episodes = range(len(training_stats['episode_rewards']))
    plt.plot(episodes, training_stats['episode_rewards'], alpha=0.6, color='blue', label='Episode Reward')
    plt.plot(episodes, training_stats['avg_rewards'], color='red', linewidth=2, label='100-Episode Average')
    plt.axhline(y=-100, color='green', linestyle='--', label='Solved Threshold (-100)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 2)
    plt.plot(training_stats['episode_lengths'], alpha=0.6, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 3)
    plt.plot(training_stats['epsilon_history'], color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 4)
    if training_stats['training_losses']:
        plt.plot(training_stats['training_losses'], alpha=0.6, color='purple')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 5)
    plt.hist(training_stats['episode_rewards'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(training_stats['episode_rewards']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(training_stats["episode_rewards"]):.1f}')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 6)
    plt.bar(range(1, len(evaluation_rewards) + 1), evaluation_rewards, color='lightgreen')
    plt.axhline(y=np.mean(evaluation_rewards), color='red', linestyle='--', 
                label=f'Average: {np.mean(evaluation_rewards):.1f}')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Reward')
    plt.title('Evaluation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dqn_training_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== DQN Acrobot Project (Slightly Larger Net) ===\n")
    agent, training_stats = train_dqn(env_name="Acrobot-v1", n_episodes=800)
    evaluation_rewards = evaluate_agent(agent, env_name="Acrobot-v1", n_episodes=20)
    create_visualizations(training_stats, evaluation_rewards)
    print("\n=== Training Summary ===")
    print(f"Final epsilon: {training_stats['final_epsilon']:.4f}")
    print(f"Best average reward: {training_stats['best_avg_reward']:.2f}")
    print(f"Final average reward: {training_stats['final_avg_reward']:.2f}")
    print(f"Average evaluation reward: {np.mean(evaluation_rewards):.2f}")
    print(f"Acrobot solved (avg reward >= -100): {training_stats['final_avg_reward'] >= -100}")
    print(f"\nFiles saved:")
    print("- dqn_training_visualization.png: Training plots")

if __name__ == "__main__":
    main()