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

from stable_baselines3 import PPO

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

        ##########
        # REWARD #
        ##########
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)
        
        self.action_space = Discrete(self._gym_env.action_space.n)

        self.setup_observations()
        self.setup_actions()

        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                        low=self._gym_env.observation_space.low,
                                        high=self._gym_env.observation_space.high)
        
        self.action_space = Discrete(self._gym_env.action_space.n)

    def setup_observations(self):
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = gym_compat.BoxGymObsSpace(self._g2op_env.observation_space)

    def setup_actions(self):
        self._gym_env.action_space.close()
        self._gym_env.action_space = gym_compat.DiscreteActSpace(self._g2op_env.action_space)

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
    project="RL Assignment PPO",
    config=config,
    sync_tensorboard=True
    )

    model = PPO(
        config["policy_type"], 
        env,
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

