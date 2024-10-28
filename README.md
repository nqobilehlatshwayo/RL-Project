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

1. Clone the repository:
```bash
git clone [repository-url]
cd power-grid-control
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Switch to desired algorithm branch:
```bash
# For DQN implementation
git checkout dqn

# For PPO implementation
git checkout ppo
```

## Environment
The project uses the Grid2Op environment ("l2rpn_case14_sandbox") with the following key features:
- Complex network topology modifications
- Generator setpoint adjustments
- Load shedding capabilities
- Real-world operational constraints
- N-1 criterion compliance requirements

## Algorithm Configurations

### DQN Parameters
- Learning Rate: 0.0001
- Buffer Size: 1,000,000
- Gamma: 0.99
- Target Update Interval: 10,000
- Initial ε: 1.0
- Final ε: 0.05
- Exploration Fraction: 0.1

### PPO Parameters
- Neural Network: [64, 64] hidden layers
- Activation: ReLU
- Learning Rate: 3e-4
- Clip Range: 0.2
- Value Function Coefficient: 0.5
- Entropy Coefficient: 0.01

## Results Summary
Performance metrics comparison between final versions of both algorithms:

| Metric | Final DQN | Final PPO |
|--------|-----------|-----------|
| Average Reward | 16.41 | 794.63 |
| Final Loss | 0.13 | 0.056 |
| Training Time | 49m 44s | 49m 33s |

## Usage
Each algorithm branch contains specific instructions for running the implementation. Generally:

1. Navigate to the desired branch
2. Run the corresponding script or improvement:

## License
[Specify your license here]

## Acknowledgments
- Grid2Op platform developers
- University of the Witwatersrand, School of Computer Science and Applied Mathematics
