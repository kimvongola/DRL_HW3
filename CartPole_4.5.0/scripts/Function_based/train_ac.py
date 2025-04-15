"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.AC import Actor_Critic

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# from omni.isaac.lab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 21
    action_range = [-20.0,20.0]  # [min, max]
    learning_rate = 0.005
    n_episodes = 5000
    initial_epsilon = 1.0
    epsilon_decay = 0.998  # reduce exploration over time
    final_epsilon = 0.05
    discount = 0.99
    buffer_size = 10000
    batch_size = 64
    hidden_dim=64


    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("device: ", device)
    def random_scaled_tensor(value_range):
        rand_tensor = torch.rand((1, 1))  
        scaled_tensor = (rand_tensor * 2 - 1) * value_range
        return scaled_tensor
    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "A2C"

    agent = Actor_Critic(
        # device = device,
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        # discretize_state_weight=discretize_state_weight,
        # hidden_dim = hidden_dim,
        initial_epsilon = initial_epsilon,

        epsilon_decay = epsilon_decay,
        final_epsilon = final_epsilon,
        discount_factor = discount,
        buffer_size = buffer_size,
        batch_size=batch_size
    )
    log_dir = os.path.join("runs", f"{Algorithm_name}_{task_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    writer = SummaryWriter(log_dir)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    sum_reward=0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        # with torch.inference_mode():
        # with torch.inference_mode():
        
        for episode in tqdm(range(n_episodes)):
            obs, _ = env.reset()
            done = False
            cumulative_reward = 0
            step=0
            # agent.learn(env,500)

            while not done:
                # ----- ACTOR step -----
                state = obs['policy']

                scaled_action, log_prob,one_hot = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(scaled_action)
                # next_obs, reward, terminated, truncated, _ = env.step(scaled_val)

                next_state = next_obs['policy']
                next_scaled_action, next_log_prob,next_one_hot = agent.select_action(next_state)

                reward_value = reward.item()
                done_flag = terminated.item() or truncated.item()

 
                agent.memory.add(state, scaled_action, reward_value, next_state, terminated.item())
                # ----- CRITIC estimates -----
                # value = agent.critic(state,one_hot)
                # next_value = agent.critic(next_state,next_one_hot)

                # ----- Compute advantage -----
                # advantage = reward_value + agent.discount_factor * next_value - value

                # ----- Update networks -----
                agent.update_policy()
                critic_loss, actor_loss = agent.update_policy()
                agent.update_target_networks()

                obs = next_obs
                cumulative_reward += reward_value
                step += 1
                done = done_flag

            # Logging
            sum_reward += cumulative_reward
            writer.add_scalar("Reward/Episode", cumulative_reward, episode)
            writer.add_scalar("Length/Episode", step, episode)
            writer.add_scalar("Policy/Epsilon", agent.epsilon, episode)

            if critic_loss is not None and actor_loss is not None:
                writer.add_scalar("Loss/Critic", critic_loss, episode)
                writer.add_scalar("Loss/Actor", actor_loss, episode)
            if episode % 100 == 0:
                print("avg_score:", sum_reward / 100.0)
                sum_reward = 0
                print(agent.epsilon)
                w_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.pth"
                full_path = os.path.join(f"weights/{task_name}", Algorithm_name)
                os.makedirs(full_path, exist_ok=True)
                agent.save_w(full_path, w_file)

            agent.decay_epsilon()

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        
        print("!!! Training is complete !!!")
        # agent.plot_durations(show_result=True)
        # plt.ioff()
        # plt.show()
        break
    # ==================================================================== #

    # close the simulator
    writer.close()
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()