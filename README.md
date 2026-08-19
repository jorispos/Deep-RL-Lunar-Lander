# Lunar Lander with Reinforcement Learning 🚀

A 2024 University of Groningen group project comparing three reinforcement learning approaches on Gymnasium's `LunarLander-v2` environment.

> **Archive note**
>
> This README was refreshed in 2026 to document the project more clearly. The implementation and results remain from the original coursework.

## 🧠 Approaches

- **Deep Q-network** with two hidden layers, experience replay, and epsilon-greedy exploration
- **Linear Q approximator** as a simpler baseline with the same replay and exploration setup
- **SARSA** with a discretized state representation and scheduled exploration

The agent observes eight state variables and chooses between four discrete engine actions. The DQN maps the state through 256- and 128-unit hidden layers to one Q-value per action.

## 📊 What We Found

The saved 350-episode runs show the DQN remaining considerably more stable than the linear baseline. Neither approach consistently solved the environment, which made the exercise a useful comparison of representation capacity and training stability rather than a benchmark result.

| Approach | Final 100-episode average return |
| --- | ---: |
| Deep Q-network | -109.1 |
| Linear Q approximator | -615.1 |

The project was informed by [Solving The Lunar Lander Problem under Uncertainty using Reinforcement Learning](https://arxiv.org/abs/2011.11850).

## 🛠️ Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install "gymnasium[box2d]==0.29.1" numpy==1.26.4 tensorflow==2.15.0
python lunar_lander_DQN.py
```

Training saves model checkpoints and episode returns every 50 episodes. Rendering is enabled periodically so the policy can be inspected while it learns.

## 📁 Repository Guide

- `lunar_lander_DQN.py` contains the deep Q-learning implementation
- `lunar_lander_Linear.py` contains the linear baseline
- `lunar_lander_sarsa.py` contains the tabular SARSA implementation
- `Data/` contains episode returns from the original runs
- `saved_models/` contains the corresponding model checkpoints

## 👥 Team

Built by [Joris Postmus](https://github.com/jorispos) and [Leon Tanis](https://github.com/Tanis1304) for the University of Groningen Reinforcement Learning Practical.
