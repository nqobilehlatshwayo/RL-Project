# Power Grid Control Using Reinforcement Learning

This repository contains implementations of Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms for power grid control using the Grid2Op environment.

## Authors
- Ayanda Thwala (2434602)
- Kholofelo Lekala (2481434)
- Thapelo Duma (2493083)
- Nqobile Hlatshwayo (2438280)

## Project Overview
This project implements and compares two reinforcement learning approaches for power grid control. The main objectives are:
- Implementing and comparing DQN and PPO algorithms
- Developing systematic improvements to both algorithms
- Analyzing the effectiveness of different approaches in maintaining grid stability

## Setup and Installation

1. Install dependencies:
```bash
pip install gymnasium grid2op lightsim2grid stable_baselines3 wandb sb3-contrib
```

## Environment
The project uses the Grid2Op environment ("l2rpn_case14_sandbox") with the following key features:
- Complex network topology modifications
- Generator setpoint adjustments
- Load shedding capabilities
- Real-world operational constraints
- N-1 criterion compliance requirements


## Usage
Each algorithm branch contains the code for running the implementation. Generally:

1. Navigate to the desired algorithm folder
2. Run the corresponding script or improvement, for example:
```bash
   python Baseline.py
```

## Results Summary
Performance metrics comparison between final versions of both algorithms:

| Metric | Final DQN | Final PPO |
|--------|-----------|-----------|
| Average Reward | 16.41 | 794.63 |
| Final Loss | 0.13 | 0.056 |
| Training Time | 49m 44s | 49m 33s |

```
 
