from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt

class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, n_observations, hidden_size, n_actions, dropout=0.0):
        super(DQN_network, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(n_observations, hidden_size)
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output layer
        self.output = nn.Linear(hidden_size, n_actions)
        # Dropout (optional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input state tensor.

        Returns:
            Tensor: Q-value estimates for each action.
        """
        x = F.relu(self.fc1(x))       # First hidden layer
        x = self.dropout(x)
        x = F.relu(self.fc2(x))       # Second hidden layer
        x = self.dropout(x)
        q_values = self.output(x)     # Output layer
        return q_values

class DQN(BaseAlgorithm):
    def __init__(
            self,
            
            num_of_action: int,# = 2,
            action_range: list,# = [-2.5, 2.5],
            hidden_dim: int,# = 64,
            learning_rate: float,# = 0.01,
            initial_epsilon: float,# = 1.0,
            epsilon_decay: float,# = 1e-3,
            final_epsilon: float,# = 0.001,
            discount_factor: float,# = 0.95,
            buffer_size: int,# = 1000,
            batch_size: int,# = 1,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            tau: float = 0.005,
            dropout: float = 0.2,
            n_observations: int = 4,


    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.device = device
        self.steps_done = 0
        self.num_of_action = num_of_action
        self.tau = tau

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        self.episode_durations = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,  
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        if np.random.rand() < self.epsilon:
            # Exploration
            action_idx = np.random.randint(self.num_of_action)
        else:
            # Exploitation
            # self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state)  # Add batch dimension: (1, state_dim)


            action_idx = torch.argmax(q_values).item()      # Choose best action
        # self.policy_net.train()  # optional
        print(action_idx)
        action_tensor = self.scale_action(action_idx)  # Map index to real value
        return action_tensor, action_idx
        # ====================================== #

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Computes the loss for policy optimization.

        Args:
            non_final_mask (Tensor): Mask indicating which states are non-final.
            non_final_next_states (Tensor): The next states that are not terminal.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of actions taken.
            reward_batch (Tensor): Batch of received rewards.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Initialize next state values to 0
        next_state_values = torch.zeros(state_batch.size(0), 1, device=self.device)
        if non_final_next_states.dim() == 3:
            print(non_final_next_states)
            non_final_next_states = non_final_next_states.squeeze(1)

        # Compute Q(s', a') for non-final next states
        with torch.no_grad():
            max_next_q_values = self.target_net(non_final_next_states).max(1, keepdim=True)[0]
            next_state_values[non_final_mask] = max_next_q_values

        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.discount_factor * next_state_values)

        # Loss = MSE(Q_estimate, Q_target)
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        return loss
        # ====================================== #

    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - non_final_mask (Tensor): A boolean mask indicating which states are non-final.
                - non_final_next_states (Tensor): The next states that are not terminal.
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
        """
        # Ensure there are enough samples in memory before proceeding
        # ========= put your code here ========= #
        # Sample a batch from memory
        if len(self.memory) < batch_size:
            return None  # Not enough data to sample
        batch = self.memory.sample()
        # ====================================== #

        # Sample a batch from memory
        states, actions, rewards, next_states, dones = batch
        # แปลงเป็น Tensor
        state_batch = torch.stack(states).to(self.device)
        # print(state_batch.shape)
        if state_batch.dim() == 3:
            state_batch = state_batch.squeeze(1)
        actions = [a for a in actions]  # flatten + convert to int
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        # if action_batch>20 or action_batch<0:
        #     print(action_batch)

        # action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)

        # reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        # สร้าง mask สำหรับ non-final state
        non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool).to(self.device)

        non_final_next_states = torch.stack(
            [s for s, d in zip(next_states, dones) if not d]
        ).to(self.device)
        non_final_next_states=non_final_next_states.squeeze(1)
        # print("non_final_next_states shape:", non_final_next_states.shape)
        # print("device:", non_final_next_states.device)

        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch
        # ========= put your code here ========= #
        
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # Generate a sample batch
        if len(self.memory) < self.batch_size:
            return  # wait until enough data
        
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        # Compute loss
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        # Perform gradient descent step
        # ========= put your code here ========= #
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        # ====================================== #

    def update_target_networks(self):
        """
        Soft update of target network weights using Polyak averaging.
        """
        # Retrieve the state dictionaries (weights) of both networks
        # ========= put your code here ========= #



        policy_params = self.policy_net.state_dict()
        target_params = self.target_net.state_dict()
        # ====================================== #
        
        # Apply the soft update rule to each parameter in the target network
        # ========= put your code here ========= #
        for key in policy_params:
            target_params[key] = self.tau * policy_params[key] + (1.0 - self.tau) * target_params[key].clone()
        # ====================================== #
        
        # Load the updated weights into the target network
        # ========= put your code here ========= #
        self.target_net.load_state_dict(target_params)

        # ====================================== #

    def learn(self, env):
        """
        Train the agent on a single step.

        Args:
            env: The environment to train in.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        state, _ = env.reset()                 # เริ่มต้น environment
        total_reward = 0.0                     # สะสม reward
        done = False                           # ตัวแปรสำหรับจบ episode
        timestep = 0                           # นับจำนวน timestep
        # ====================================== #

        while not done:
            # Predict action from the policy network
            # ========= put your code here ========= #
            action = self.select_action(state)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            self.memory.add(state, action, reward, next_state, done)
            # ====================================== #

            # Update state

            # Perform one step of the optimization (on the policy network)
            self.update_policy()

            # Soft update of the target network's weights
            self.update_target_networks()

            timestep += 1
            if done:
                self.plot_durations(timestep)
                break
        # ====================================== #

    # Consider modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #