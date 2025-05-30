import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import os

# Parámetros del entorno
GRID_SIZE = (5, 5)  # (filas, columnas)
OBSTACLES = [(1, 1), (1, 2), (2, 2), (3, 4), (5, 1), (5, 2), (5, 3), (4, 5), (3, 2), (4, 2)]
START = (0, 0)
GOAL = (4, 4)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Entorno personalizado
class GridWorldEnv:
    def __init__(self, grid_size, start, goal, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.reset()

    def reset(self):
        self.agent_pos = list(self.start)
        return self._get_state()

    def _get_state(self):
        return self.agent_pos[0] * self.grid_size[1] + self.agent_pos[1]  # posición linealizada

    def step(self, action):
        old_pos = self.agent_pos.copy()

        if action == 0:   # UP
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1: # DOWN
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size[0] - 1)
        elif action == 2: # LEFT
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3: # RIGHT
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size[1] - 1)

        if tuple(self.agent_pos) in self.obstacles:
            self.agent_pos = old_pos  # no se mueve si choca
            reward = -1
            done = False
        elif tuple(self.agent_pos) == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.1
            done = False

        return self._get_state(), reward, done

# Red neuronal
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Entrenamiento
def train():
    env = GridWorldEnv(GRID_SIZE, START, GOAL, OBSTACLES)
    state_size = GRID_SIZE[0] * GRID_SIZE[1]
    action_size = len(ACTIONS)
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = deque(maxlen=2000)

    episodes = 1000
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(100):
            state_tensor = torch.zeros(state_size)
            state_tensor[state] = 1.0

            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            next_state_tensor = torch.zeros(state_size)
            next_state_tensor[next_state] = 1.0

            memory.append((state_tensor, action, reward, next_state_tensor, done))
            state = next_state
            total_reward += reward

            if done:
                break

        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)
            for state_tensor, action, reward, next_state_tensor, done in minibatch:
                target = reward
                if not done:
                    with torch.no_grad():
                        target += gamma * torch.max(model(next_state_tensor)).item()
                target_f = model(state_tensor)
                target_f = target_f.clone()
                target_f[action] = target
                optimizer.zero_grad()
                output = model(state_tensor)
                loss = criterion(output, target_f)
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (e+1) % 100 == 0:
            print(f"Episodio {e+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return model, env

def render_env(env, agent_pos):
    os.system('cls' if os.name == 'nt' else 'clear')
    for i in range(env.grid_size[0]):
        row = ""
        for j in range(env.grid_size[1]):
            pos = (i, j)
            if pos in env.obstacles:
                row += " X "
            elif pos == env.goal:
                row += " G "
            elif pos == tuple(agent_pos):
                row += " R "
            else:
                row += " . "
        print(row)
    time.sleep(0.3)

def run_simulation(model, env):
    state = env.reset()
    total_reward = 0
    action_sequence = []

    for _ in range(50):
        state_tensor = torch.zeros(GRID_SIZE[0] * GRID_SIZE[1])
        state_tensor[state] = 1.0

        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        action_sequence.append(ACTIONS[action])
        render_env(env, env.agent_pos)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

        if done:
            render_env(env, env.agent_pos)
            print("¡Meta alcanzada!")
            break
    else:
        print("No llegó a la meta.")

    print(f"\nRecompensa total: {total_reward:.2f}")
    print("Secuencia de movimientos:")
    print(action_sequence)
    return action_sequence


# Al final del main
if __name__ == "__main__":
    model, env = train()
    run_simulation(model, env)