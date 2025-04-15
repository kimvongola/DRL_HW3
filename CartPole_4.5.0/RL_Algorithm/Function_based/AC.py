import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Actor, self).__init__()

        # ========= put your code here ========= #
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)  # Convert logits to probability distribution
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Selected action values.
        """
        # ========= put your code here ========= #
        return self.model(state)
        # ====================================== #

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        """
        Critic network for Q-value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of hidden units in layers.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        # ========= put your code here ========= #
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output Q-value (scalar)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state, action):
        """
        Forward pass for Q-value estimation.

        Args:
            state (Tensor): Current state of the environment.
            action (Tensor): Action taken by the agent.

        Returns:
            Tensor: Estimated Q-value.
        """
        # ========= put your code here ========= #\
        if action.dim() == 1:
            action = action.unsqueeze(0)  # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        return self.model(x)
        # ====================================== #

class Actor_Critic(BaseAlgorithm):
    def __init__(self, 
                num_of_action: int,# = 2,
                action_range: list,# = [-2.5, 2.5],
                discount_factor: float,# = 0.95,
                learning_rate: float,# = 0.01,
                initial_epsilon: float,# = 1.0,
                epsilon_decay: float,# = 1e-3,
                final_epsilon: float,# = 0.001,

                buffer_size: int,# = 256,
                batch_size: int,#= 1,
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                hidden_dim = 64,
                dropout = 0.02, 
                tau: float = 0.005,
                n_observations: int = 4,

                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.actor_target = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)
        self.critic_target = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)

        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor

        self.update_target_networks(tau=1)  # initialize target networks


        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

        super(Actor_Critic, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,  
        )

    def select_action(self, state, noise=0.0):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
            state (Tensor): The current state of the environment.
            noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - scaled_action (Tensor): The final action formatted for environment interaction.
                - clipped_action (Tensor): The selected action probability (or log-prob if needed).
        """

        # self.actor.eval()
        with torch.no_grad():
            action_probs = self.actor(state.unsqueeze(0)).squeeze(0)  # [num_actions]
        # self.actor.train()
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        one_hot = torch.nn.functional.one_hot(action, num_classes=self.num_of_action).float()
        # Apply scaling if environment expects continuous actions
        scaled_action = self.scale_action(action.item())
        print(scaled_action)
        # scaled_action = action.view(1, 1).to(state.device)
        
        return scaled_action, log_prob,one_hot
        # ====================================== #
    
    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
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
        # Convert to list of ints if it's a list of tensors
        actions = [a.item() if isinstance(a, torch.Tensor) else int(a) for a in actions]
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)


        # reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Convert next_states and dones to tensors
        next_state_batch = torch.stack(next_states).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # If next_state_batch has an extra dimension, squeeze it
        if next_state_batch.dim() == 3:
            next_state_batch = next_state_batch.squeeze(1)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Computes the loss for policy optimization.

        Args:
            - states (Tensor): The batch of current states.
            - actions (Tensor): The batch of actions taken.
            - rewards (Tensor): The batch of rewards received.
            - next_states (Tensor): The batch of next states received.
            - dones (Tensor): The batch of dones received.

        Returns:
            Tensor: Computed critic & actor loss.
        """
        # ========= put your code here ========= #
        # Update Critic
        with torch.no_grad():
            next_action_probs = self.actor_target(next_states)
            next_action_samples = torch.distributions.Categorical(next_action_probs).sample()
            next_action_onehot = torch.nn.functional.one_hot(next_action_samples, num_classes=self.num_of_action).float()
            target_q = self.critic_target(next_states, next_action_onehot)
            target_value = rewards + self.discount_factor * target_q * (1 - dones)
        # ------- Critic Loss -------
        action_onehot = torch.nn.functional.one_hot(actions.squeeze(-1), num_classes=self.num_of_action).float()
        current_q = self.critic(states, action_onehot)
        critic_loss = nn.functional.mse_loss(current_q, target_value)
        # ------- Advantage Calculation -------
        advantage = (target_value - current_q).detach()

        # ------- Actor Loss -------
        predicted_actions = self.actor(states)
        dist = torch.distributions.Categorical(predicted_actions)
        log_probs = dist.log_prob(actions.squeeze(-1))
        actor_loss = (-log_probs * advantage.squeeze()).mean()

        return critic_loss, actor_loss
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #

        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return None, None
        states, actions, rewards, next_states, dones = sample

        # Calculate losses
        critic_loss, actor_loss = self.calculate_loss(states, actions, rewards, next_states, dones)

        # ------- Update Critic -------
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic.optimizer.step()

        # ------- Update Actor -------
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor.optimizer.step()

        return critic_loss.item(), actor_loss.item()
        # ====================================== #


    def update_target_networks(self, tau=None):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        # ========= put your code here ========= #
        tau = tau if tau is not None else self.tau
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        # ====================================== #

    def learn(self, env, max_steps, num_agents, noise_scale=0.1, noise_decay=0.99):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
            num_agents (int): Number of agents in the environment.
            noise_scale (float, optional): Initial exploration noise level. Defaults to 0.1.
            noise_decay (float, optional): Factor by which noise decreases per step. Defaults to 0.99.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        state, _ = env.reset()
        state = torch.tensor(state['policy'][0], dtype=torch.float32).to(self.device).unsqueeze(0)

        total_reward = 0
        done = False
        step = 0
        # ====================================== #

        for step in range(max_steps):
            # Predict action from the policy network
            # ========= put your code here ========= #
            action_logits = self.actor(state)
            dist = torch.distributions.Categorical(action_logits)
            action_idx = dist.sample()

            # Add exploration noise
            noisy_action_idx = torch.clamp(action_idx + int(np.random.normal(0, noise_scale)), 0, self.num_of_action - 1)
            action = self.scale_action(noisy_action_idx.item()).to(self.device)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state['policy'][0], dtype=torch.float32).to(self.device).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
            done = terminated or truncated
            done_tensor = torch.tensor([done], dtype=torch.float32).to(self.device)
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            # Parallel Agents Training
            if num_agents > 1:
                pass
            # Single Agent Training
            else:
                self.memory.add(state, action_idx.unsqueeze(0), reward, next_state, done_tensor)
            # ====================================== #

            # Update state
            state = next_state
            total_reward += reward.item()

            # Decay the noise to gradually shift from exploration to exploitation
            noise_scale *= noise_decay

            # Perform one step of the optimization (on the policy network)
            self.update_policy()

            # Update target networks
            self.update_target_networks()

            if done:
                break

        return total_reward
