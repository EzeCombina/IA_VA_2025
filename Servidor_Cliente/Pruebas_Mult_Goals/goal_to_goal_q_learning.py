import socket
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

# ---------- ENTORNO GRIDWORLD ----------
class GridWorldEnv:
    def __init__(self, grid_size, start, goals, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.goals = goals
        self.obstacles = obstacles
        self.reset()

    def reset(self, start=None, goal=None):
        self.agent_pos = list(start if start else self.start)
        self.goal = goal if goal else random.choice(self.goals)
        return self._get_state()

    def _get_state(self):
        state = torch.zeros(self.grid_size[0] * self.grid_size[1] * 2)
        agent_index = self.agent_pos[0] * self.grid_size[1] + self.agent_pos[1]
        goal_index = self.goal[0] * self.grid_size[1] + self.goal[1]
        state[agent_index] = 1.0
        state[self.grid_size[0] * self.grid_size[1] + goal_index] = 1.0
        return state

    def step(self, action):
        old_pos = self.agent_pos.copy()
        if action == 0:
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size[0] - 1)
        elif action == 2:
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size[1] - 1)

        if tuple(self.agent_pos) in self.obstacles:
            self.agent_pos = old_pos
            reward = -1
            done = False
        elif tuple(self.agent_pos) == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.1
            done = False

        return self._get_state(), reward, done

# ---------- MODELO ----------
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

# ---------- ENTRENAMIENTO ----------
def train(env, state_size, action_size):
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = deque(maxlen=3000)

    episodes = 600
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64

    meta_pairs = [(env.start, g) for g in env.goals]
    meta_pairs += [(g1, g2) for g1 in env.goals for g2 in env.goals if g1 != g2]
    pair_index = 0

    for e in range(episodes):
        start, goal = meta_pairs[pair_index]
        pair_index = (pair_index + 1) % len(meta_pairs)

        state = env.reset(start=start, goal=goal)
        total_reward = 0

        for _ in range(100):
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if done:
                break

        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)
            for s, a, r, s_next, d in minibatch:
                target = r
                if not d:
                    with torch.no_grad():
                        target += gamma * torch.max(model(s_next)).item()
                target_f = model(s).clone()
                target_f[a] = target
                optimizer.zero_grad()
                loss = criterion(model(s), target_f)
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (e + 1) % 100 == 0:
            print(f"[Entrenamiento] Episodio {e + 1}, Recompensa total: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return model

# ---------- SIMULACIÓN ----------
def run_simulation(model, env, actions_str):
    paths = {}

    pairs = [(env.start, g) for g in env.goals]
    pairs += [(g1, g2) for g1 in env.goals for g2 in env.goals if g1 != g2]

    for start, goal in pairs:
        state = env.reset(start=start, goal=goal)
        path = []
        total_reward = 0

        for _ in range(50):
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()

            path.append(actions_str[action])
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break

        key = f"{start}->{goal}"
        paths[key] = {
            "path": path,
            "reward": total_reward
        }

    return paths

# ---------- SOCKET CLIENTE ----------
def main():
    HOST = '127.0.0.1'
    PORT = 65432
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    try:
        while True:
            datos = client_socket.recv(4096)
            if not datos:
                print("[CLIENTE] Servidor desconectado.")
                break

            config = json.loads(datos.decode())
            GRID_SIZE = tuple(config["GRID_SIZE"])
            OBSTACLES = [tuple(ob) for ob in config["OBSTACLES"]]
            START = tuple(config["START"])
            GOALS = [tuple(g) for g in config["GOALS"]]
            ACTIONS = config["ACTIONS"]

            env = GridWorldEnv(GRID_SIZE, START, GOALS, OBSTACLES)
            state_size = GRID_SIZE[0] * GRID_SIZE[1] * 2
            action_size = len(ACTIONS)

            model = train(env, state_size, action_size)
            resultados = run_simulation(model, env, ACTIONS)

            print("\n======= Caminos desde el punto de inicio original =======")
            for g in GOALS:
                key = f"{START}->{g}"
                if key in resultados:
                    print(f"Meta {g}:")
                    print(f"  Secuencia: {resultados[key]['path']}")
                    print(f"  Recompensa total: {resultados[key]['reward']:.2f}\n")

            print("======= Caminos entre metas =======")
            for g1 in GOALS:
                for g2 in GOALS:
                    if g1 != g2:
                        key = f"{g1}->{g2}"
                        if key in resultados:
                            print(f"{g1} -> {g2}:")
                            print(f"  Secuencia: {resultados[key]['path']}")
                            print(f"  Recompensa total: {resultados[key]['reward']:.2f}\n")

            # Mostrar la grilla
            print("\n--- Estado del entorno ---")
            for i in range(GRID_SIZE[0]):
                fila = ""
                for j in range(GRID_SIZE[1]):
                    pos = (i, j)
                    if pos == START:
                        fila += " R "
                    elif pos in OBSTACLES:
                        fila += " x "
                    elif pos in GOALS:
                        fila += " G "
                    else:
                        fila += " . "
                print(fila)

            client_socket.send(json.dumps(resultados).encode())

    except KeyboardInterrupt:
        print("[CLIENTE] Finalizado por el usuario.")
    finally:
        client_socket.close()

# ---------- EJECUCIÓN ----------
if __name__ == "__main__":
    main()

