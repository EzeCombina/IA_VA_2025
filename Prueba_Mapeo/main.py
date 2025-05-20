import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Parámetros ===
robot_size_cm = 5  # Tamaño del robot en cm
image_path = "aula4.png"  # Reemplazá con el nombre de tu imagen
room_size_cm = (500, 500)  # Tamaño real del ambiente (ancho x alto) en cm

# === Cargar imagen ===
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === Binarizar (suelo blanco, obstáculos negros) ===
# Umbral automático usando Otsu
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Mostrar imagen binarizada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title("Binarizada: suelo vs obstáculos")

# === Calcular escala y tamaño de celda en píxeles ===
img_height, img_width = binary.shape
scale_x = room_size_cm[0] / img_width
scale_y = room_size_cm[1] / img_height
pixels_per_cell_x = int(robot_size_cm / scale_x)
pixels_per_cell_y = int(robot_size_cm / scale_y)

# === Crear matriz de ocupación ===
grid = []
for y in range(0, img_height, pixels_per_cell_y):
    row = []
    for x in range(0, img_width, pixels_per_cell_x):
        cell = binary[y:y+pixels_per_cell_y, x:x+pixels_per_cell_x]
        # Obstáculo si al menos 30% de la celda es negra
        obstacle_ratio = np.sum(cell == 0) / cell.size
        row.append(1 if obstacle_ratio > 0.3 else 0)
    grid.append(row)

# Convertir a array numpy para visualización
grid = np.array(grid)

# === Mostrar imagen mapeada ===
plt.subplot(1, 2, 2)
plt.imshow(grid, cmap='Greys')
plt.title("Mapa tipo matriz (0: libre, 1: obstáculo)")
plt.show()
