# Swarm Path Planning with Q-Learning

## Table of Contents

1. [Project Title](#project-title)
2. [Project Description](#project-description)
3. [How to Install and Run the Project](#how-to-install-and-run-the-project)
4. [How to Use the Project](#how-to-use-the-project)

---

## Project Title

**Swarm Path Planning with Q-Learning** – A simulation where multiple agents navigate a grid with obstacles to reach a shared goal using **swarm intelligence rules** combined with **Q-learning**.

---

## Project Description

This project demonstrates how a group of agents (a **swarm**) can cooperate to reach a goal in a grid environment while avoiding obstacles and collisions.

It combines:

- **Swarm Intelligence**

  - Goal attraction (agents move toward the target).
  - Obstacle avoidance.
  - Agent separation (prevent clustering or collisions).

- **Q-Learning**
  - Reinforcement learning that allows agents to improve their decision-making over time.
  - Balances exploration (try new moves) and exploitation (use learned strategies).
  - Smart termination logic prevents infinite loops by detecting when agents are stuck.

### Why this project?

- To explore **multi-agent coordination** in dynamic environments.
- To demonstrate how **reinforcement learning** can enhance traditional rule-based swarm behaviors.
- To showcase **visual simulations** with Matplotlib for better understanding of agent behavior.

### Challenges & Future Improvements

- Current reward function is simple and may not scale well to larger grids.
- Agents sometimes take longer paths if swarm bias conflicts with Q-learning choices.
- Future improvements could include:
  - Smarter state representation (e.g., including nearby agents/obstacles).
  - Training with larger agent swarms.
  - Adding alternative reinforcement learning methods (e.g., Deep Q-Networks).

---

## How to Install and Run the Project

### Requirements

- Python 3.x
- Libraries:
  ```bash
  pip install numpy matplotlib
  ```

---

## How to Use the Project

1. Run the simulation:

- Run swarmPathPlanning.py.
  - python swarmPathPlanning.py
- Agents start in the bottom-left of the grid and attempt to reach a goal in the top-right corner.

2. Watch the behavior:

- Agents move step by step, balancing swarm rules and Q-learning decisions.
- Obstacles block paths and require detours.
- Agents at the goal are allowed to “stack” without collisions.

3. Smart Termination:

- The simulation continues until all agents reach the goal.
- If stuck, exploration/learning rates are automatically adjusted.

### Visualization

- Colored circles = agents.
- Gray rectangles = obstacles.
- Gold circle = goal.
- Dashed lines = agent paths.
