import gymnasium as gym
import numpy as np
from gymnasium import spaces

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.PlotGrid import PlotMatplot

from lightsim2grid import LightSimBackend
from sb3_contrib import RecurrentPPO  # Changed from PPO to RecurrentPPO
import matplotlib.pyplot as plt

import wandb

class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)
        
        self.setup_observations()
        self.setup_actions()

        self._max_steps = 100  # Set max steps
        self._current_step = 0

    def setup_observations(self):
        """Setup observation space for a selected subset of the grid state."""
        
        # Define the observation subset keys
        self.obs_keys = ["gen_p", "gen_q", "gen_v",
                        "load_p", "load_q","load_v", 
                        "p_or",
                        "timestep_overflow",
                        "topo_vect",
                        "rho",
                        "line_status"]
        
        # Initialize lists to store the lower and upper bounds for the selected observation subset
        low = []
        high = []
        
        # Get the full observation space from the original environment
        obs_space = self._gym_env.observation_space
        
        # Loop through the desired observation keys and collect bounds for the subset
        for key in self.obs_keys:
            space = obs_space[key]
            if isinstance(space, spaces.Box):
                low.extend(space.low)
                high.extend(space.high)
            elif isinstance(space, spaces.Discrete):
                low.append(0)
                high.append(space.n - 1)
            elif isinstance(space, spaces.MultiBinary):
                low.extend([0] * space.n)
                high.extend([1] * space.n)
            else:
                raise ValueError(f"Unsupported observation space: {type(space)}")

        # Create a Box space for the observation subset
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32)
        )

    def setup_actions(self):
        act_space = self._gym_env.action_space
        
        # Modify act_keys to only include specific actions you're working with
        self.act_keys = ["set_line_status",
                        "set_bus", 
                        "redispatch"]
        
        # Collect action spaces for the specific keys
        self.act_spaces = [act_space[key] for key in self.act_keys]

        n_dim = 0
        for space in self.act_spaces:
            if isinstance(space, spaces.Discrete):
                n_dim += space.n
            elif isinstance(space, spaces.Box):
                n_dim += np.prod(space.shape)
            elif isinstance(space, spaces.MultiBinary):
                n_dim += space.n
            else:
                raise ValueError(f"Unsupported action space: {type(space)}")

        # Define the continuous action space for the PPO agent
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(n_dim,),
            dtype=np.float32
        )

    def _flatten_obs(self, obs):
        """Enhanced observation flattening with normalization"""
        flattened = []
        
        for key in self.obs_keys:
            if key not in obs:
                continue
                
            value = obs[key]
            if isinstance(value, np.ndarray):
                # Normalize continuous values
                if value.dtype in [np.float32, np.float64]:
                    max_val = np.abs(value).max()
                    if max_val > 0:
                        value = value / max_val
                flattened.extend(value.flatten())
            elif isinstance(value, (int, float)):
                # Normalize scalar values
                if isinstance(value, float):
                    value = np.clip(value / 100.0, -1, 1)  # Assuming reasonable range
                flattened.append(value)
        
        return np.array(flattened, dtype=np.float32)
    
    def _continuous_to_mixed(self, action):
        """Enhanced action conversion with better scaling"""
        mixed_action = {}
        idx = 0
        
        for key, space in zip(self.act_keys, self.act_spaces):
            if isinstance(space, spaces.Discrete):
                # Improved discrete action selection
                n_actions = space.n
                action_slice = action[idx:idx + n_actions]
                # Use softmax-like normalization for better action selection
                action_probs = np.exp(action_slice) / np.sum(np.exp(action_slice))
                discrete_value = np.argmax(action_probs)
                mixed_action[key] = discrete_value
                idx += n_actions
            
            elif isinstance(space, spaces.Box):
                # Better continuous action scaling
                box_shape = np.prod(space.shape)
                continuous_action = action[idx:idx + box_shape]
                # Scale to actual action space
                low, high = space.low, space.high
                scaled_action = low + (continuous_action + 1.0) * 0.5 * (high - low)
                mixed_action[key] = scaled_action
                idx += box_shape
            
            elif isinstance(space, spaces.MultiBinary):
                # Improved binary action conversion
                binary_action = np.where(action[idx:idx + space.n] > 0, 1, 0)
                mixed_action[key] = binary_action
                idx += space.n
        
        return mixed_action

    def reset(self, seed=None):
        obs, info = self._gym_env.reset(seed=seed, options=None)
        self._current_step = 0  # Reset the step counter at the beginning of each episode
        return self._flatten_obs(obs), info

    def step(self, action):
        mixed_action = self._continuous_to_mixed(action)
        obs, reward, terminated, truncated, info = self._gym_env.step(mixed_action)

        # Increment step counter
        self._current_step += 1

        # Check if max steps reached
        if self._current_step >= self._max_steps:
            truncated = True  # End the episode due to reaching max steps

        return self._flatten_obs(obs), reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the grid state and visualize CompleteObservation attributes"""
        obs = self._g2op_env.get_obs()  # Get CompleteObservation instance
        # Render observation details here (based on your requirements)
        return obs  # Return observation if needed elsewhere


def train_recurrent_ppo(env, total_timesteps=20000):
    model = RecurrentPPO(
        "MlpLstmPolicy",  # Use LSTM policy for RecurrentPPO
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
        )

    model.learn(total_timesteps=total_timesteps)
    return model


# def plot_rewards(all_episode_rewards):
#     plt.figure(figsize=(10, 6))
#     for episode, rewards in enumerate(all_episode_rewards):
#         steps = range(1, len(rewards) + 1)
#         plt.plot(steps, rewards, label=f'Episode {episode + 1}')
    
#     plt.title('Reward Over Time Per Episode')
#     plt.xlabel('Step')
#     plt.ylabel('Reward')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def evaluate_and_plot(env, model, n_episodes=5):
    """
    Evaluate the model and create plots without GUI dependencies
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    all_episode_rewards = []
    all_episode_steps = []
    
    print("\n=== Running Test Episodes ===")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_rewards = []
        lstm_states = None
        done = False
        step = 0
        
        print(f"\nStarting Episode {episode + 1}")
        
        while not done:
            # Get action from model with LSTM states
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            step += 1
            done = terminated or truncated
            
            if step % 10 == 0:  # Print progress every 10 steps
                print(f"Episode {episode + 1} - Step {step}: Current Reward = {reward:.2f}, "
                      f"Total = {sum(episode_rewards):.2f}")
        
        all_episode_rewards.append(episode_rewards)
        all_episode_steps.append(step)
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {sum(episode_rewards):.2f}")
        print(f"Average Reward per Step: {sum(episode_rewards)/step:.2f}")
        
        if info.get("exception", None) is not None:
            print(f"Exception occurred: {info['exception']}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cumulative Rewards
    for episode, rewards in enumerate(all_episode_rewards):
        cumulative_rewards = np.cumsum(rewards)
        ax1.plot(range(1, len(rewards) + 1), cumulative_rewards, 
                label=f'Episode {episode + 1}')
    
    ax1.set_title('Cumulative Reward Over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cumulative Reward')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Per-step Rewards
    for episode, rewards in enumerate(all_episode_rewards):
        ax2.plot(range(1, len(rewards) + 1), rewards, 
                label=f'Episode {episode + 1}', alpha=0.7)
    
    ax2.set_title('Reward per Step')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('recurrent_ppo_evaluation3.png')
    print("\nPlot saved as 'recurrent_ppo_evaluation3.png'")
    plt.close()
    
    # Return statistics for further analysis if needed
    return {
        'all_episode_rewards': all_episode_rewards,
        'all_episode_steps': all_episode_steps,
        'average_total_reward': np.mean([sum(rewards) for rewards in all_episode_rewards]),
        'average_episode_length': np.mean(all_episode_steps)
    }

def main():
    env = Gym2OpEnv()

    print("\n=== Environment Spaces ===")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    
    print("\n=== Starting RecurrentPPO Training ===")
    model = train_recurrent_ppo(env, total_timesteps=20000)
    
    model.save("recurrent_ppo_grid_model")
    print("\nModel saved as 'recurrent_ppo_grid_model'")

    # Use the new evaluation function
    stats = evaluate_and_plot(env, model)
    
    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    print(f"Average Total Reward: {stats['average_total_reward']:.2f}")
    print(f"Average Episode Length: {stats['average_episode_length']:.2f} steps")

if __name__ == "__main__":
    main()
