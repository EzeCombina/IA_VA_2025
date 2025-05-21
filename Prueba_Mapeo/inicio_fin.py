import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

# === Parámetros ===
robot_size_cm = 5
image_path = "aula4.png"
room_size_cm = (500, 500)

# === Cargar imagen ===
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en '{image_path}'")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img_height, img_width = binary.shape
scale_x = room_size_cm[0] / img_width
scale_y = room_size_cm[1] / img_height
pixels_per_cell_x = int(robot_size_cm / scale_x)
pixels_per_cell_y = int(robot_size_cm / scale_y)

# === Variables globales ===
start_pixel_color = None
selected_points = []
obstacle_color = None

# === Obtener tamaño de la grilla anticipadamente ===
temp_grid = []
for y in range(0, img_height, pixels_per_cell_y):
    row = []
    for x in range(0, img_width, pixels_per_cell_x):
        row.append(0)
    temp_grid.append(row)
temp_grid = np.array(temp_grid)
grid_rows, grid_cols = temp_grid.shape

# === Callback de clic para seleccionar puntos ===
def on_click(event):
    global start_pixel_color, obstacle_color

    if event.inaxes:
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        col = x
        row = y

        if row >= grid_rows or col >= grid_cols or row < 0 or col < 0:
            print("⚠️ Punto fuera de la grilla.")
            return

        px_y = row * pixels_per_cell_y
        px_x = col * pixels_per_cell_x

        # Evitar que se salga del borde
        px_y = min(px_y, img_height - pixels_per_cell_y)
        px_x = min(px_x, img_width - pixels_per_cell_x)

        local_cell = binary[px_y:px_y+pixels_per_cell_y, px_x:px_x+pixels_per_cell_x]
        local_avg = np.mean(local_cell)
        start_pixel_color = 255 if local_avg > 127 else 0
        obstacle_color = 0 if start_pixel_color == 255 else 255

        print(f"[INFO] Color de inicio: {'blanco' if start_pixel_color == 255 else 'negro'} → Obstáculos = {'negro' if obstacle_color == 0 else 'blanco'}")

        selected_points.append((row, col))
        ax.plot(col, row, 'ro' if len(selected_points) == 1 else 'go')
        plt.draw()

        if len(selected_points) == 2:
            print(f"Inicio: {selected_points[0]}, Fin: {selected_points[1]}")
            fig.canvas.mpl_disconnect(cid)

# === Mostrar imagen para clics ===
fig, ax = plt.subplots()
ax.imshow(binary, cmap='gray')
plt.title("Seleccioná INICIO (clic 1) y FIN (clic 2)")
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

if len(selected_points) != 2:
    raise ValueError("No se seleccionaron ambos puntos correctamente.")

start, goal = selected_points

# === Generar la grilla final con obstáculos ===
grid = []
for y in range(0, img_height, pixels_per_cell_y):
    row = []
    for x in range(0, img_width, pixels_per_cell_x):
        cell = binary[y:y+pixels_per_cell_y, x:x+pixels_per_cell_x]
        obstacle_ratio = np.sum(cell == obstacle_color) / cell.size
        row.append(1 if obstacle_ratio > 0.3 else 0)
    grid.append(row)
grid = np.array(grid)

# === A* ===
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(array, start, goal):
    neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_heap = []
    heapq.heappush(open_heap, (fscore[start], start))

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0]+i, current[1]+j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]:
                if array[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in open_heap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (fscore[neighbor], neighbor))
    return None

# === Ejecutar planificación y simular ===
path = astar(grid, start, goal)
if path is None:
    print("[ERROR] No se encontró un camino.")
else:
    print(f"[INFO] Camino encontrado con {len(path)} pasos.")
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Greys')
    ax.plot(start[1], start[0], 'ro', label='Inicio')
    ax.plot(goal[1], goal[0], 'go', label='Fin')
    plt.title("Simulación del movimiento")

    for pos in path:
        if pos != start and pos != goal:
            ax.plot(pos[1], pos[0], 'bo')  # Azul = trayectoria
            plt.pause(0.05)

    plt.legend()
    plt.show()



