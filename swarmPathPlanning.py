import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

plt.switch_backend('TkAgg')

@dataclass
class Config:
    grid_size: Tuple[int, int] = (12, 12)
    num_agents: int = 4
    num_obstacles: int = 2
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.3


class Position:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y
    
    def __iter__(self):
        return iter((self.x, self.y))
    
    def distance(self, other: 'Position') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def add(self, dx: int, dy: int) -> 'Position':
        return Position(self.x + dx, self.y + dy)
    
    def tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


class Environment:
    def __init__(self, config: Config):
        self.config = config
        self.width, self.height = config.grid_size
        self.grid = np.zeros(config.grid_size)
        self.obstacles = []
        self.goal = Position(self.width - 2, self.height - 2)
        
        self._place_obstacles()
        self.grid[self.goal.x, self.goal.y] = 2
    
    def _place_obstacles(self):
        for _ in range(self.config.num_obstacles):
            for attempt in range(50):
                w, h = random.randint(2, 3), random.randint(2, 3)
                x = random.randint(2, self.width - w - 2)
                y = random.randint(2, self.height - h - 2)
                
                if not np.any(self.grid[x:x+w, y:y+h] == 1):
                    self.grid[x:x+w, y:y+h] = 1
                    self.obstacles.append((x, y, w, h))
                    break
    
    def is_valid(self, pos: Position) -> bool:
        return (0 <= pos.x < self.width and 0 <= pos.y < self.height and 
                self.grid[pos.x, pos.y] != 1)


class Agent:
    def __init__(self, agent_id: int, pos: Position, color: str):
        self.id = agent_id
        self.pos = pos
        self.color = color
        self.path = [pos]
        self.at_goal = False
    
    def move(self, new_pos: Position):
        self.pos = new_pos
        self.path.append(new_pos)


class SwarmIntelligence:
    def __init__(self):
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # UP, DOWN, LEFT, RIGHT, STAY
    
    def get_direction(self, agent: Agent, env: Environment, agents: List[Agent]) -> Tuple[int, int]:
        pos = agent.pos
        
        # Goal attraction
        goal_dx = 1 if env.goal.x > pos.x else (-1 if env.goal.x < pos.x else 0)
        goal_dy = 1 if env.goal.y > pos.y else (-1 if env.goal.y < pos.y else 0)
        
        # Obstacle repulsion
        obs_dx = obs_dy = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_pos = pos.add(dx, dy)
                if not env.is_valid(check_pos):
                    obs_dx -= dx
                    obs_dy -= dy
        
        # Agent separation
        agent_dx = agent_dy = 0
        for other in agents:
            if other.id != agent.id and other.pos.tuple() != env.goal.tuple():
                dist = pos.distance(other.pos)
                if 0 < dist < 2:
                    agent_dx += 1 if pos.x > other.pos.x else -1
                    agent_dy += 1 if pos.y > other.pos.y else -1
        
        total_dx = goal_dx + obs_dx + agent_dx
        total_dy = goal_dy + obs_dy + agent_dy
        
        # Convert to action index
        if abs(total_dx) > abs(total_dy):
            return 1 if total_dx > 0 else 0
        elif total_dy != 0:
            return 3 if total_dy > 0 else 2
        return 4


class QLearning:
    def __init__(self, config: Config, num_agents: int):
        self.config = config
        self.q_tables = [defaultdict(lambda: defaultdict(float)) for _ in range(num_agents)]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    
    def get_state(self, agent_pos: Position, goal_pos: Position) -> Tuple[int, int]:
        dx = min(3, max(-3, goal_pos.x - agent_pos.x))
        dy = min(3, max(-3, goal_pos.y - agent_pos.y))
        return (dx, dy)
    
    def choose_action(self, agent_id: int, state: Tuple[int, int], 
                     swarm_action: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return swarm_action if random.random() < 0.7 else random.randint(0, 4)
        
        q_values = [self.q_tables[agent_id][state][a] for a in range(5)]
        if all(q == 0 for q in q_values):
            return swarm_action
        
        q_values[swarm_action] += 1.0  # Swarm bias
        return q_values.index(max(q_values))
    
    def update(self, agent_id: int, state: Tuple[int, int], action: int, 
              reward: float, next_state: Tuple[int, int]):
        """Update Q-table."""
        current_q = self.q_tables[agent_id][state][action]
        next_q_max = max([self.q_tables[agent_id][next_state][a] for a in range(5)], default=0)
        new_q = current_q + self.config.learning_rate * (
            reward + self.config.discount_factor * next_q_max - current_q)
        self.q_tables[agent_id][state][action] = new_q


class Visualizer:
    def __init__(self, env: Environment):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    def draw(self, agents: List[Agent], step: int, epsilon: float):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.env.width - 0.5)
        self.ax.set_ylim(-0.5, self.env.height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for x, y, w, h in self.env.obstacles:
            rect = patches.Rectangle((x-0.4, y-0.4), w-0.2, h-0.2, 
                                   facecolor='gray', alpha=0.7)
            self.ax.add_patch(rect)
        
        # Draw goal
        goal_circle = patches.Circle(self.env.goal.tuple(), 0.3, color='gold', alpha=0.8)
        self.ax.add_patch(goal_circle)
        self.ax.text(self.env.goal.x, self.env.goal.y, 'GOAL', 
                    ha='center', va='center', fontweight='bold')
        
        # Draw agent paths
        for agent in agents:
            if len(agent.path) > 1:
                path_x, path_y = zip(*[p.tuple() for p in agent.path])
                self.ax.plot(path_x, path_y, color=agent.color, alpha=0.5, 
                           linewidth=2, linestyle='--')
        
        # Draw agents
        agents_at_goal = [a for a in agents if a.at_goal]
        for agent in agents:
            x, y = agent.pos.tuple()
            
            if agent.at_goal and len(agents_at_goal) > 1:
                angle = 2 * np.pi * agents_at_goal.index(agent) / len(agents_at_goal)
                offset_x = 0.15 * np.cos(angle)
                offset_y = 0.15 * np.sin(angle)
                circle = patches.Circle((x + offset_x, y + offset_y), 0.15, 
                                      color=agent.color, alpha=0.9)
            else:
                circle = patches.Circle((x, y), 0.2, color=agent.color, alpha=0.9)
            
            self.ax.add_patch(circle)
            self.ax.text(x, y, str(agent.id + 1), ha='center', va='center',
                        fontweight='bold', color='white', fontsize=8)
        
        # Title
        agents_at_goal_count = len(agents_at_goal)
        self.ax.set_title(f'Step {step} | {agents_at_goal_count}/{len(agents)} at goal | ε={epsilon:.3f}')
        plt.draw()
        plt.pause(0.05)


class SwarmPathPlanning:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.env = Environment(self.config)
        self.swarm = SwarmIntelligence()
        self.qlearning = QLearning(self.config, self.config.num_agents)
        self.viz = Visualizer(self.env)
        
        # Initialize agents
        self.agents = []
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i in range(self.config.num_agents):
            pos = self._find_start_position()
            agent = Agent(i, pos, colors[i % len(colors)])
            self.agents.append(agent)
        
        # Progress tracking
        self.epsilon = self.config.epsilon
        self.steps_without_progress = 0
        self.last_agents_at_goal = 0
    
    def _find_start_position(self) -> Position:
        for _ in range(50):
            x = random.randint(0, self.env.width // 3)
            y = random.randint(0, self.env.height // 3)
            pos = Position(x, y)
            
            if (self.env.is_valid(pos) and 
                all(pos.distance(a.pos) >= 2 for a in self.agents)):
                return pos
        return Position(0, 0)
    
    def _calculate_reward(self, old_pos: Position, new_pos: Position, valid: bool) -> float:
        if new_pos.tuple() == self.env.goal.tuple():
            return 100
        if not valid:
            return -20
        
        old_dist = old_pos.distance(self.env.goal)
        new_dist = new_pos.distance(self.env.goal)
        
        if new_dist < old_dist:
            return 10
        elif new_dist > old_dist:
            return -5
        return -1
    
    def _is_valid_move(self, agent: Agent, new_pos: Position) -> bool:
        if not self.env.is_valid(new_pos):
            return False
        if new_pos.tuple() == self.env.goal.tuple():  # Multiple agents can be at goal
            return True
        return not any(other.pos.tuple() == new_pos.tuple() 
                      for other in self.agents if other.id != agent.id)
    
    def step(self):
        for agent in self.agents:
            agent.at_goal = (agent.pos.tuple() == self.env.goal.tuple())
            if agent.at_goal:
                continue
            
            # Get swarm direction
            swarm_action = self.swarm.get_direction(agent, self.env, self.agents)
            
            # Get Q-learning action
            state = self.qlearning.get_state(agent.pos, self.env.goal)
            action = self.qlearning.choose_action(agent.id, state, swarm_action, self.epsilon)
            
            # Execute move
            old_pos = agent.pos
            dx, dy = self.swarm.actions[action]
            new_pos = old_pos.add(dx, dy)
            
            valid = self._is_valid_move(agent, new_pos)
            if valid:
                agent.move(new_pos)
            
            # Update Q-learning
            reward = self._calculate_reward(old_pos, new_pos, valid)
            next_state = self.qlearning.get_state(agent.pos, self.env.goal)
            self.qlearning.update(agent.id, state, action, reward, next_state)
    
    def _check_progress(self) -> bool:
        agents_at_goal = sum(1 for a in self.agents if a.at_goal)
        
        if agents_at_goal > self.last_agents_at_goal:
            self.steps_without_progress = 0
            self.last_agents_at_goal = agents_at_goal
        else:
            self.steps_without_progress += 1
        
        if self.steps_without_progress > 200:
            self.epsilon = min(0.8, self.epsilon + 0.2)
            self.steps_without_progress = 0
            print("Boosting exploration...")
        
        return self.steps_without_progress < 600  # Ultimate failsafe
    
    def run(self, show_every: int = 20) -> Tuple[int, bool]:
        print("Swarm Path Planning Simulation")
        print("=" * 40)
        print(f"Agents: {self.config.num_agents} | Grid: {self.config.grid_size}")
        print(f"Goal: {self.env.goal.tuple()}")
        print("=" * 40)
        
        plt.ion()
        step = 0
        
        while not all(a.at_goal for a in self.agents):
            self.step()
            step += 1
            
            if not self._check_progress():
                print(f"Simulation terminated at step {step} (infinite loop prevention)")
                break
            
            if step % show_every == 0 or step < 100:
                self.viz.draw(self.agents, step, self.epsilon)
                agents_at_goal = sum(1 for a in self.agents if a.at_goal)
                print(f"Step {step:3d}: {agents_at_goal}/{self.config.num_agents} at goal")
            
            if step % 100 == 0:
                self.epsilon = max(0.05, self.epsilon * 0.95)
        
        # Final results
        self.viz.draw(self.agents, step, self.epsilon)
        plt.ioff()
        
        success = all(a.at_goal for a in self.agents)
        agents_at_goal = sum(1 for a in self.agents if a.at_goal)
        
        print(f"\nRESULTS:")
        print(f"{'SUCCESS' if success else 'PARTIAL'}: {agents_at_goal}/{self.config.num_agents} agents at goal")
        print(f"Total steps: {step}")
        for agent in self.agents:
            print(f"  Agent {agent.id + 1}: {len(agent.path) - 1} moves")
        
        plt.show(block=True)
        return step, success

# Main execution
if __name__ == "__main__":
    config = Config(
        grid_size=(12, 12),
        num_agents=4,
        num_obstacles=2
    )
    
    sim = SwarmPathPlanning(config)
    final_step, success = sim.run(show_every=20)