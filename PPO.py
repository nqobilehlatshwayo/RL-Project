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
        self.obs_keys = list(obs_space.spaces.keys())

        low = []
        high = []
        for space in obs_space.values():
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
        self.act_keys = list(act_space.spaces.keys())
        self.act_spaces = list(act_space.values())

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

        self.action_space = spaces.Box(low=-1, high=1, shape=(n_dim,), dtype=np.float32)

    def reset(self, seed=None):
        obs, info = self._gym_env.reset(seed=seed)
        return self._flatten_obs(obs), info

    def step(self, action):
        mixed_action = self._continuous_to_mixed(action)
        obs, reward, terminated, truncated, info = self._gym_env.step(mixed_action)
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
            if isinstance(space, spaces.Discrete):
                mixed_action[key] = np.argmax(action[idx:idx+space.n])
                idx += space.n
            elif isinstance(space, spaces.Box):
                n = np.prod(space.shape)
                mixed_action[key] = action[idx:idx+n].reshape(space.shape)
                idx += n
            elif isinstance(space, spaces.MultiBinary):
                mixed_action[key] = (action[idx:idx+space.n] > 0).astype(int)
                idx += space.n
        return mixed_action

def train_ppo(env, total_timesteps=10000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model

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
    model = train_ppo(vec_env, total_timesteps=10000)
    print("PPO training completed.\n")

    # Visualize steps with the trained agent using the unwrapped environment
    max_steps = 100
    curr_step = 0
    curr_return = 0
    is_done = False

    obs, info = env.reset()  # Use the unwrapped environment for visualization
    print(f"step = {curr_step} (reset):")
    print(f"\t obs = {obs}")
    print(f"\t info = {info}\n\n")

    while not is_done and curr_step < max_steps:
        action, _states = model.predict(obs[np.newaxis, :], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])  # Use the unwrapped environment

        curr_step += 1
        curr_return += reward
        is_done = terminated or truncated

        print(f"step = {curr_step}: ")
        print(f"\t obs = {obs}")
        print(f"\t reward = {reward}")
        print(f"\t terminated = {terminated}")
        print(f"\t truncated = {truncated}")
        print(f"\t info = {info}")

        is_action_valid = not (info.get("is_illegal", False) or info.get("is_ambiguous", False))
        print(f"\t is action valid = {is_action_valid}")
        if not is_action_valid:
            print(f"\t\t reason = {info.get('exception', 'Unknown')}")
        print("\n")

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print("###########")

if __name__ == "__main__":
    main()