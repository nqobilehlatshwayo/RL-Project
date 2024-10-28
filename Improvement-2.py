import gymnasium as gym
from gymnasium.spaces import Discrete, Box

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import DQN
from stable_baselines3.common.prioritized_replay_buffer import PrioritizedReplayBuffer

# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    
    def __init__(self):

        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)
        
        self.action_space = Discrete(self._gym_env.action_space.n)

    def setup_observations(self):
        obs_attr_to_keep = ["rho", "p_or", "gen_p", "load_p", "line_status", "target_dispatch", "actual_dispatch"]
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = gym_compat.BoxGymObsSpace(self._g2op_env.observation_space,
                                                         attr_to_keep=obs_attr_to_keep)

    def setup_actions(self):
        act_attr_to_keep = ["set_line_status", "set_bus", "redispatch"]
        self._gym_env.action_space.close()
        self._gym_env.action_space = gym_compat.DiscreteActSpace(self._g2op_env.action_space,
                                                          attr_to_keep=act_attr_to_keep)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
       return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()
    
def main():

    env = Gym2OpEnv()

    env.render()

    config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 500000,
    "env_id": "l2rpn_case14_sandbox",
    }


    run = wandb.init(
    project="RL Assignment DQN",
    config=config,
    sync_tensorboard=True
    )

    model = DQN(
        config["policy_type"], 
        env,
        replay_buffer_class=PrioritizedReplayBuffer,
        verbose=2,
        device="cuda",
        tensorboard_log=f"runs/{run.id}, learning_rate=0.0001"
        )

    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(),
                    )
    run.finish()

    env.render()

if __name__ == "__main__":
    main()