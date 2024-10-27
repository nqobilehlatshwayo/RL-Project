import gymnasium as gym
import numpy as np
from gymnasium import spaces

import grid2op
from grid2op import gym_compat 
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        obs_space = self._gym_env.observation_space
        self.obs_keys = list([
            'topo_vect', 'rho', 
            'actual_dispatch'
        ])

        low = []
        high = []
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

        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32)
        )

    def setup_actions(self):
        act_space = self._gym_env.action_space

        self.act_keys = list(["set_bus", "change_line_status", "redispatch"])
        self.act_spaces = []
        for key in self.act_keys:
            self.act_spaces.append(act_space[key])

        low = []
        high = []
        for key, space in zip(self.act_keys, self.act_spaces):
            if key == "set_bus":
                low.extend([x + 1 for x in space.low])
                high.extend(space.high)
            elif key == "change_line_status":
                low.extend([0] * space.n)
                high.extend([1] * space.n)
            else:
                low.extend(space.low)
                high.extend(space.high)

        self.action_space = spaces.Box(low=np.array(low, dtype=np.float32),
                                       high=np.array(high, dtype=np.float32))

    def reset(self, seed=None):
        obs, info = self._gym_env.reset(seed=seed)
        obs = self.obs_limits(obs)
        return self._flatten_obs(obs), info

    def step(self, action):
        mixed_action = self._continuous_to_mixed(action)
        obs, reward, terminated, truncated, info = self._gym_env.step(mixed_action)
        obs = self.obs_limits(obs)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def _flatten_obs(self, obs):
        flattened = []
        for key in self.obs_keys:
            value = obs[key]
            if isinstance(value, np.ndarray):
                flattened.extend(value.flatten())
            elif isinstance(value, (int, float)):
                flattened.append(value)
            else:
                raise ValueError(f"Unsupported observation type: {type(value)}")
        return np.array(flattened, dtype=np.float32)

    def _continuous_to_mixed(self, action):
        mixed_action = {}
        idx = 0
        for key, space in zip(self.act_keys, self.act_spaces):
            if key == "set_bus":
                n = np.prod(space.shape)
                mixed_action[key] = np.round( action[idx:idx+n].reshape(space.shape))
                idx += n
            elif key == "change_line_status":
                np_arr = np.array(action[idx:idx+space.n])
                largest_indices = np.argpartition(np_arr, -4)[-4:]
                result = np.zeros_like(np_arr, dtype=int)
                for i in largest_indices:
                    if np_arr[i] >= 0.5:
                        result[i] = 1
                mixed_action[key] = result.tolist()
                idx += space.n
            elif key == "redispatch":
                n = np.prod(space.shape)
                mixed_action[key] = []
                aa = 0
                for a in action[idx:idx+n].reshape(space.shape):
                    a = max(a, -self.limits['gen_margin_down'][aa])
                    a = min(a, self.limits['gen_margin_up'][aa])
                    mixed_action[key].append(a)
                    aa += 1
                idx += n
            elif key == "curtail":
                n = np.prod(space.shape)
                mixed_action[key] = action[idx:idx+n].reshape(space.shape)
                idx += n
        return mixed_action
    
    def obs_limits(self, all_obs):
        obs = {}
        self.limits = all_obs
        for key, value in all_obs.items():
            if key in self.obs_keys:
                obs[key] = value
        return obs

def train_ppo(env, total_timesteps=10000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate(env, model):
    # Visualize steps with the trained agent using the unwrapped environment
    max_steps = 100
    curr_step = 0
    curr_return = 0
    is_done = False

    rewards = []

    obs, info = env.reset()  # Use the unwrapped environment for visualization

    while not is_done and curr_step < max_steps:
        action, _states = model.predict(obs[np.newaxis, :], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])  # Use the unwrapped environment

        curr_step += 1
        curr_return += reward
        rewards.append(reward)
        is_done = terminated or truncated

    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print(f"average return = {curr_return/curr_step} \n")
    return curr_return, curr_step, rewards

def main():
    env = Gym2OpEnv()

    print("#####################")
    print("# OBSERVATION SPACE #")
    print("#####################")
    print(env.observation_space)
    print("#####################\n")

    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    print(env.action_space)
    print("#####################\n\n")

    # Wrap the environment for training
    vec_env = DummyVecEnv([lambda: env])

    # Train the agent
    model = train_ppo(vec_env, total_timesteps=50000)
    print("PPO training completed.\n")

    curr_return, curr_step, total = 0, 0, 100
    ave_rewards = [0]*total
    for ep in range(total):
        print(f"episode = {ep}")
        r, s, rewards = evaluate(env, model)
        curr_return += r
        curr_step += s
        for i in range(len(rewards)):
            ave_rewards[i] += rewards[i]/total
    
    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return/total}")
    print(f"total steps = {curr_step/total}")
    print(f"return per step = {curr_return/curr_step}")
    print("###########")

    plt.plot(ave_rewards)
    plt.xlabel('Step')
    plt.ylabel('Average Rewards')
    plt.title('Training Rewards Over Time')
    plt.show()
    plt.savefig('PPO_improvement_1.png')

    
if __name__ == "__main__":
    main()
