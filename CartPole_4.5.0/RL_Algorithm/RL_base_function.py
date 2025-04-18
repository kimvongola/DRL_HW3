import numpy as np
from collections import defaultdict, namedtuple, deque
import random
from enum import Enum
import os
import json
import torch
import torch.nn as nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size = 1):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences the buffer can hold.
            batch_size (int): Number of experiences to sample per batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience to the replay buffer.

        Args:
            state (Tensor): The current state of the environment.
            action (Tensor): The action taken at this state.
            reward (Tensor): The reward received after taking the action.
            next_state (Tensor): The next state resulting from the action.
            done (bool): Whether the episode has terminated.
        """
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            - state_batch: Batch of states.
            - action_batch: Batch of actions.
            - reward_batch: Batch of rewards.
            - next_state_batch: Batch of next states.
            - done_batch: Batch of terminal state flags.
        """
        batch = random.sample(self.memory, k=self.batch_size)

        # แยก component แต่ละอันออกจาก tuple
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of stored experiences.
        """
        return len(self.memory)


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,

        num_of_action: int, #= 2,
        action_range: list, #= [-2.0, 2.0],
        learning_rate: float ,#= 1e-3,
        initial_epsilon: float ,#= 1.0,
        epsilon_decay: float, #= 1e-3,
        final_epsilon: float ,#= 0.001,
        discount_factor: float, #= 0.95,
        buffer_size: int ,#= 1000,
        batch_size: int #= 1,  
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        self.w = np.zeros((4, num_of_action))
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def q(self, obs, a=None):
        """Returns the linearly-estimated Q-value for a given state and action."""
        # ========= put your code here ========= #
        # Extract the policy tensor from the dictionary
        state_features = obs['policy'].detach().cpu().numpy().astype(np.float32).flatten()

    
        # Compute Q-values
        if a is not None:
            return float(np.dot(state_features, self.w[:, a]))

        else:
            return np.dot(state_features, self.w)  # returns shape (num_actions,)


        # ====================================== #
        
    
    def scale_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n].
            n (int): Number of discrete actions (inclusive range from 0 to n).
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """

        # ========= put your code here ========= #
        # # ใช้ตัวแปรจาก __init__
        action_min, action_max = self.action_range
        # n = self.num_of_action

        # # แปลง action index → normalized [0, 1]
        # normalized = action / (n - 1)

        # # แปลงเป็นค่าต่อเนื่องในช่วง [action_min, action_max]
        # continuous_action = action_min + normalized * (action_max - action_min)
        action_min,action_max=self.action_range
        continuous_action = action_min + (action / (self.num_of_action - 1)) * (action_max - action_min)

        return torch.tensor([[continuous_action]], dtype=torch.float32) 
        pass
        # ====================================== #
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

        # ====================================== #

    def save_w(self, path, filename):
        """
        Save weight parameters to a JSON file.
        """
        os.makedirs(path, exist_ok=True)
        
        # Ensure it ends in .json
        if not filename.endswith(".json"):
            filename += ".json"

        full_path = os.path.join(path, filename)

        # Convert NumPy array to nested list
        weights_list = self.w.tolist()

        # Save to JSON
        with open(full_path, 'w') as f:
            json.dump(weights_list, f, indent=4)

        print(f"Saved weights to {full_path}")
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        full_path = os.path.join(path, filename)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Weight file not found: {full_path}")

        self.w = np.load(full_path)
        print(f"Loaded weights from {full_path}")
        # ====================================== #