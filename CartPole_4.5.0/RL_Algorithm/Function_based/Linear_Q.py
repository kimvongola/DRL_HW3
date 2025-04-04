from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
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
            # Linear function approximation for Q-values
        feature_vector=super.discretize_state(obs)
        next_feature_vector=super.discretize_state(next_obs)
        current_q_value = self.weights.dot(feature_vector)  # Dot product of weights and feature vector for the current state-action pair
        
        # Calculate the next Q-value using the next state (and next action if SARSA)
        next_q_value = self.weights.dot(next_feature_vector) if not terminated else 0
        
        # Calculate TD error
        td_error = reward + self.discount_factor * next_q_value - current_q_value
        
        # Update the weights using the learning rate
        self.w += self.learning_rate * td_error * feature_vector

        self.training_error.append(td_error)
        pass
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
    if np.random.rand() < self.epsilon:
        # Exploration: Choose a random action
        action = np.random.choice(self.num_of_action)
    else:
        # Exploitation: Choose the best action (highest Q-value)
        action_values = self.w.dot(state)
        action = np.argmax(action_values)
        return action
        pass
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
            # ===== Initialize trajectory collection variables ===== #
        total_reward = 0  # Track total reward accumulated in the episode
        done = False  # Flag to indicate if the episode has terminated
        steps = 0  # Step counter
        # Reset environment to get initial state
        state, _ = env.reset()  # Assuming reset() returns a state and info
    
        # ===== Main loop for each episode ===== #
        while not done and steps < max_steps:
            # Select an action based on the epsilon-greedy policy
            action = self.select_action(state)
    
            # Take the action and observe the next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
    
            # Update the agent's knowledge (Q-values or weights)
            self.update(state, action, reward, next_state, terminated)
    
            # Update total reward and step counter
            total_reward += reward
            steps += 1
    
            # If the episode ends (either terminated or truncated), set done to True
            done = terminated or truncated
    
            # Move to the next state
            state = next_state
    
        # Optionally: track or return the total reward for logging or analysis
        return total_reward
        pass
        # ====================================== #
    




    
