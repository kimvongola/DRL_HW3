from __future__ import annotations
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    def __init__(
            self,

            num_of_action: int, #= 2,
            action_range: list ,#= [-2.5, 2.5],
            learning_rate: float ,#= 0.01,
            initial_epsilon: float, #= 1.0,
            epsilon_decay: float, #= 1e-3,
            final_epsilon: float, #= 0.001,
            discount_factor: float, #= 0.95,
            buffer_size: int ,#= 1000,
            batch_size: int #= 1,  
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

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size
         
        )
        
    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action: int,
        terminated: bool
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            obs (dict): The current state observation, containing feature representations.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs (dict): The next state observation.
            next_action (int): The action taken in the next state (used in SARSA).
            terminated (bool): Whether the episode has ended.
        """
        # ========= put your code here ========= #

        # Get the feature vector for the current state
        feature_vector = obs['policy'].detach().cpu().numpy().astype(np.float32).flatten()
        feature_vector /= (np.linalg.norm(feature_vector) + 1e-8)

        # Get the Q-value for the next state-action pair
        next_feature_vector = next_obs['policy'].detach().cpu().numpy().astype(np.float32)
        
        # Get current Q-value for the selected action
        current_q_values = self.q(obs, action)


        # Calculate next Q-value (use max if not terminated, otherwise 0)
        next_q_value = self.q(next_obs)
        if terminated:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(next_q_value)
        # Calculate the TD error
        td_error = target-current_q_values

        # td_error = np.clip(td_error, -0.5, 0.5)



        self.w[:, action] += self.lr * td_error * feature_vector  # Update the weights

        # self.w = np.clip(self.w, -0.1, 0.1)
        # self.w*=0.999

        # self.training_error.append(td_error)
        
        # ====================================== #

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #

        action_values = np.dot(state, self.w)  # shape: (num_actions,)
        
        if np.random.rand() < self.epsilon:
            # Exploration
            action = np.random.randint(self.num_of_action)
        else:
            # Exploitation with uncertainty (softmax sampling)
            action=np.argmax(action_values)

        action_tensor = self.scale_action(action)
        return action_tensor, action
        # return torch.tensor(action,dtype=torch.float32)
        
        # ====================================== #

    def learn(self, env, max_steps):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        # Initialize trajectory collection variables
        total_reward = 0  # Track total reward accumulated in the episode
        done = False  # Flag to indicate if the episode has terminated
        steps = 0  # Step counter
        # Reset environment to get initial state
        state, _ = env.reset()  # Assuming reset() returns a state and info
        # Main loop for each episode
        while not done and steps < max_steps:
            # Select an action based on the epsilon-greedy policy
            discrete_state=state['policy'].detach().cpu().float()
            action,action_idx = self.select_action(discrete_state)
            # Take the action and observe the next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
    
            # Update the agent's knowledge (Q-values or weights)
            discrete_next_state=next_state['policy'].detach().cpu().float()
            next_action,next_action_idx = self.select_action(discrete_next_state)  # In case of SARSA, select the next action
            self.update(state, action_idx, reward, next_state, next_action_idx, done)

            # Update total reward and step counter
            total_reward += reward
            steps += 1
    
            # If the episode ends (either terminated or truncated), set done to True
            done = terminated or truncated
    
            # Move to the next state
            state = next_state
    
        # Optionally: track or return the total reward for logging or analysis
        return total_reward
        # ====================================== #





    