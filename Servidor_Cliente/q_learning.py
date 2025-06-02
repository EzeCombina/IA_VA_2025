import socket
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import os

# ---------- Socket: recibir configuración del servidor ----------
def main():
    # Conectar al servidor
    HOST = '127.0.0.1'
    PORT = 65432
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    try:
        while True:
            # Esperar a recibir la configuración del servidor
            datos = client_socket.recv(4096)
            if not datos:
                print("[CLIENTE] Servidor desconectado.")
                break
                
            # Procesar la configuración recibida en formato JSON
            config = json.loads(datos.decode())
            GRID_SIZE = tuple(config["GRID_SIZE"])  # (filas, columnas)
            OBSTACLES = [tuple(ob) for ob in config["OBSTACLES"]]  # Convertir a tuplas
            START = tuple(config["START"])  # Posición inicial del agente
            GOAL = tuple(config["GOAL"])  # Posición de la meta
            ACTIONS = config["ACTIONS"]  # Lista de acciones disponibles

            # Con la funcion GridWorldEnv, crear el entorno y entrenar el modelo
            env = GridWorldEnv(GRID_SIZE, START, GOAL, OBSTACLES)
            state_size = GRID_SIZE[0] * GRID_SIZE[1]
            action_size = len(ACTIONS)

            # Entrenar el modelo DQN
            model = train(env, state_size, action_size)
            # Realizar la simulación y obtener la secuencia de acciones
            action_sequence = run_simulation(model, env, GRID_SIZE, ACTIONS)

            # Enviar la secuencia de acciones al servidor
            client_socket.send(json.dumps(action_sequence).encode())
    except KeyboardInterrupt:
        print("[CLIENTE] Finalizado por el usuario.")
    finally:
        client_socket.close()

# ---------- Entorno y DQN ----------

"""
class se usa para definir clases, que son plantillas para crear objetos. 
Una clase agrupa datos (atributos) y funciones (métodos) que operan sobre esos datos, 
siguiendo el paradigma de programación orientada a objetos (OOP)
"""

class GridWorldEnv:
    # Con la funcion __init_, se inicializa la clase GridWorldEnv, que representa un entorno de cuadrícula para un agente.
    def __init__(self, grid_size, start, goal, obstacles):
        # srlf hace referencia a la instancia actual de la clase, permitiendo acceder a sus atributos y métodos.
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.reset()

    # Con la funcion reset, se reinicia el entorno, colocando al agente en su posición inicial.
    def reset(self):
        self.agent_pos = list(self.start)
        return self._get_state()

    # Con la funcion _get_state, se obtiene el estado actual del agente en la cuadrícula, representado como un número único.
    def _get_state(self):
        return self.agent_pos[0] * self.grid_size[1] + self.agent_pos[1]

    # Con la funcion step, se actualiza la posición del agente según la acción tomada,
    def step(self, action):
        old_pos = self.agent_pos.copy()

        # Las acciones se mapean a movimientos en la cuadrícula: 
        # 0 = arriba, 1 = abajo, 2 = izquierda, 3 = derecha.
        if action == 0:
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size[0] - 1)
        elif action == 2:
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size[1] - 1)

        # Verificar si el agente colisiona con un obstáculo o alcanza la meta.
        if tuple(self.agent_pos) in self.obstacles:
            self.agent_pos = old_pos
            reward = -1  # Penalización por colisión con obstáculo
            done = False
        elif tuple(self.agent_pos) == self.goal:
            reward = 10  # Recompensa por alcanzar la meta
            done = True
        else:
            reward = -0.1  # Penalización por cada paso para fomentar la eficiencia
            done = False

        return self._get_state(), reward, done

# Con la clase DQN, se define un modelo de red neuronal para el aprendizaje por refuerzo.
"""
Esta red toma un vector de entrada (estado), lo pasa por una capa 
oculta de 128 neuronas con activación ReLU y produce un vector de salida 
con tantas dimensiones como acciones posibles.
"""
class DQN(nn.Module):
    # Con la funcion __init_, se inicializa la red neuronal con una capa oculta y una capa de salida.
    # input_dim es el tamaño del estado (número de celdas en la cuadrícula) 
    # output_dim es el número de acciones posibles.
    def __init__(self, input_dim, output_dim):
        # nn.Module es la clase base para todos los módulos de redes neuronales en PyTorch.
        super(DQN, self).__init__()
        self.fc = nn.Sequential( 
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    # Con la funcion forward, se define cómo los datos pasan a través de la red neuronal.
    def forward(self, x):
        return self.fc(x)

# Con la funcion train, se entrena el modelo DQN utilizando el entorno GridWorldEnv.
def train(env, state_size, action_size):
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador Adam para actualizar los pesos de la red
    criterion = nn.MSELoss()  # Función de pérdida MSE para calcular el error entre las predicciones y los valores reales
    memory = deque(maxlen=2000)  # Memoria para almacenar experiencias pasadas (estado, acción, recompensa, siguiente estado, hecho)

    episodes = 1000         # Número de episodios para entrenar el modelo
    gamma = 0.95            # Factor de descuento para las recompensas futuras
    epsilon = 1.0           # Epsilon-greedy para la exploración: probabilidad de elegir una acción aleatoria
    epsilon_min = 0.01      # Valor mínimo de epsilon para evitar exploración excesiva
    epsilon_decay = 0.995   # Tasa de decaimiento de epsilon para reducir la exploración con el tiempo
    batch_size = 64         # Tamaño del lote para el entrenamiento por lotes

    # Entrenamiento del modelo a través de episodios
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(100):
            state_tensor = torch.zeros(state_size)  # Crear un tensor de ceros del tamaño del estado
            state_tensor[state] = 1.0

            # Epsilon-greedy: elegir una acción aleatoria con probabilidad epsilon.
            # random.random() genera un número aleatorio entre 0 y 1.
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            # Si no se elige una acción aleatoria, se utiliza el modelo para predecir la mejor acción.
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

        # Si la memoria tiene suficientes experiencias, se realiza un entrenamiento por lotes.
        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)  # Muestrear un lote aleatorio de experiencias de la memoria
            # Actualizar los pesos del modelo utilizando el lote muestreado.
            for state_tensor, action, reward, next_state_tensor, done in minibatch:
                target = reward
                # Si no se ha alcanzado el estado final, se calcula el valor objetivo utilizando la predicción del modelo.
                if not done:
                    with torch.no_grad():
                        target += gamma * torch.max(model(next_state_tensor)).item()
                # Clonar el tensor de salida del modelo para actualizar la acción seleccionada.
                target_f = model(state_tensor)
                target_f = target_f.clone()
                target_f[action] = target
                optimizer.zero_grad()
                output = model(state_tensor)
                loss = criterion(output, target_f)
                loss.backward()
                optimizer.step()

        # Reducir epsilon para disminuir la exploración con el tiempo.
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Imprimir el progreso del entrenamiento cada 100 episodios.
        if (e+1) % 100 == 0:
            print(f"Episodio {e+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return model

# ---------- Funciones de renderizado y simulación ----------
def render_env(env, agent_pos):
    # Esta función renderiza el entorno en la consola, mostrando la cuadrícula, obstáculos, meta y posición del agente.
    #os.system('cls' if os.name == 'nt' else 'clear')
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
    print("\n")
    time.sleep(0.3)

# Con la funcion run_simulation, se ejecuta una simulación del entorno utilizando el modelo entrenado.
def run_simulation(model, env, GRID_SIZE, ACTIONS):
    state = env.reset()
    total_reward = 0
    action_sequence = []

    # Se ejecuta la simulación durante 50 pasos o hasta que se alcance la meta.
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

# ---------- MAIN ----------
if __name__ == "__main__":
    main()
