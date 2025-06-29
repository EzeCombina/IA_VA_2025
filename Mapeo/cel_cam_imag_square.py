import cv2 as cv
import numpy as np
import time
import datetime

# Diccionario y detector ArUco
# Estas funciones son parte de OpenCV y permiten detectar marcadores ArUco en imágenes o videos.
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

# IDs de los ArUcos de referencia (matriz)
aruco_positions = {
    25: ('TL', 2),
    33: ('TR', 3),
    30: ('BR', 0),
    23: ('BL', 1)
}

# ID del ArUco móvil
mobile_id = 99

# Dimensiones de la matriz virtual
grid_width = 400
grid_height = 400
step = 80  # distancia entre líneas de la grilla

# Coordenadas destino fijas para la grilla (TL, TR, BR, BL)
dst_pts = np.array([
    [0, 0],
    [grid_width, 0],
    [grid_width, grid_height],
    [0, grid_height]
], dtype=np.float32)

# Colores
GRID_COLOR = (255, 0, 0)
MARKER_COLOR = (0, 255, 255)

# Cámara
#cap = cv.VideoCapture("http://192.168.206.143:8080/video")
cap = cv.VideoCapture(1)

# Definimos variables globales para el printeo de posiciones
last_print = 0
print_interval = 3  # segundos

def guardar_snapshot_con_grilla(frame_con_grilla, src_pts):
    """
    Recorta y guarda la zona delimitada por los ArUcos (con la grilla ya dibujada encima).
    """
    global dst_pts

    src_pts = np.array(src_pts, dtype=np.float32)

    # Matriz de homografía para recorte corregido
    h_crop = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Aplicamos la transformación al frame con grilla ya superpuesta
    snapshot = cv.warpPerspective(frame_con_grilla, h_crop, (grid_width, grid_height))

    # Generamos un nombre con timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.png"
    cv.imwrite(filename, snapshot)
    print(f"[INFO] Imagen guardada como {filename}")

# Bucle principal
while True:
    # Leer frame de la cámara
    ret, frame = cap.read()
    # Verificar si se leyó correctamente
    if not ret:
        break

    # Convertir a escala de grises
    corners, ids, _ = detector.detectMarkers(frame)

    # Dibujar los marcadores detectados
    if ids is not None:
        ids = ids.flatten()
        detected = {int(id): corner for id, corner in zip(ids, corners)}

        if all(id in detected for id in aruco_positions):
            src_pts = []
            # Nuevo orden: TL, TR, BR, BL
            for id in [25, 33, 30, 23]:
                _, idx = aruco_positions[id]
                pt = detected[id][0][idx]
                src_pts.append(pt)
            src_pts = np.array(src_pts, dtype=np.float32)

            # Homografía para insertar la grilla
            # Con la funcion cv.getPerspectiveTransform obtenemos la matriz de transformación
            # que mapea los puntos de la grilla a los puntos de los ArUcos detectados
            h_matrix = cv.getPerspectiveTransform(dst_pts, src_pts)
            inv_h_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)

            # Crear matriz visual transparente (4 canales BGRA)
            grid_img = np.zeros((grid_height, grid_width, 4), dtype=np.uint8)

            line_color = (255, 0, 0, 255)  # azul con alfa

            for x in range(0, grid_width, step):
                cv.line(grid_img, (x, 0), (x, grid_height), line_color, 1)
            for y in range(0, grid_height, step):
                cv.line(grid_img, (0, y), (grid_width, y), line_color, 1)

            # Proyectar grilla
            # La funcion cv.warpPerspective aplica la transformación de perspectiva a la imagen de la grilla
            warped = cv.warpPerspective(grid_img, h_matrix, (frame.shape[1], frame.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)

            # Separar canales y preparar máscara alpha
            # Esta máscara se usa para mezclar la grilla con el frame original
            # La funcion warped devuelve una imagen con 4 canales (BGRA)
            # BGRA hace referencia a los canales azul, verde, rojo y alfa (transparencia)
            warped_bgr = warped[..., :3]
            # alpha_mask es el canal alfa de la imagen warped
            alpha_mask = warped[..., 3]
            # Con la funcion merge combinamos los canales de la máscara alpha en un solo canal
            alpha_mask_3 = cv.merge([alpha_mask, alpha_mask, alpha_mask])

            # Normalizar máscara
            # Convertimos la máscara alpha a un rango de 0 a 1 para mezclar correctamente
            alpha = alpha_mask_3.astype(float) / 255.0
            frame_float = frame.astype(float)
            warped_float = warped_bgr.astype(float)

            # Mezclar imágenes
            frame = cv.convertScaleAbs(warped_float * alpha + frame_float * (1 - alpha))
            frame_con_grilla = frame.copy()

            # Detectar marcador móvil
            if mobile_id in detected:
                mobile_corners = detected[mobile_id][0]  # 4 esquinas del ArUco (en imagen original)
                
                # Dibujar el contorno del ArUco en el frame original
                pts_int = mobile_corners.astype(int)
                cv.polylines(frame, [pts_int], isClosed=True, color=MARKER_COLOR, thickness=2)
                
                # Proyectar las 4 esquinas al espacio de la grilla
                mobile_corners = mobile_corners.reshape(-1, 1, 2)
                transformed_corners = cv.perspectiveTransform(mobile_corners, inv_h_matrix)
                
                # Extraer x, y proyectados
                xs = transformed_corners[:, 0, 0]
                ys = transformed_corners[:, 0, 1]
                
                # Verificamos si TODAS las esquinas están dentro de la misma celda
                cols = xs // step
                rows = ys // step

                if np.all(cols == cols[0]) and np.all(rows == rows[0]):
                    col = int(cols[0])
                    row = int(rows[0])

                    if 0 <= col < (grid_width // step) and 0 <= row < (grid_height // step):
                        # Centro del ArUco proyectado
                        x_center = np.mean(xs)
                        y_center = np.mean(ys)

                        # Límites de la celda
                        cell_x_min = col * step
                        cell_x_max = (col + 1) * step
                        cell_y_min = row * step
                        cell_y_max = (row + 1) * step

                        # Zona central permitida (por ejemplo, 60% del área)
                        margin_ratio = 0.2  # margen de 20% en cada lado
                        margin_x = step * margin_ratio
                        margin_y = step * margin_ratio

                        # Límites de la zona central
                        central_x_min = cell_x_min + margin_x
                        central_x_max = cell_x_max - margin_x
                        central_y_min = cell_y_min + margin_y
                        central_y_max = cell_y_max - margin_y

                        # Verificar si el centro del ArUco está dentro de la zona central
                        if central_x_min <= x_center <= central_x_max and central_y_min <= y_center <= central_y_max:
                            celda = f"Fila {row}, Columna {col}"

                            # Mostrar en pantalla
                            cv.putText(frame, f"Rel pos: ({x_center:.1f}, {y_center:.1f})", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, MARKER_COLOR, 2)
                            cv.putText(frame, celda, (50, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, MARKER_COLOR, 2)

                            # Mostrar por consola cada X segundos
                            if time.time() - last_print > print_interval:
                                print(f"ArUco {mobile_id} centrado en celda: ({row}, {col})")
                                last_print = time.time()
                        else:
                            celda = "ArUco no centrado en la celda"
                    else:
                        celda = "Fuera de la grilla"
                else:
                    celda = "Marcador no completamente en una sola celda"


    cv.imshow("Tracking ArUco", frame)
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        if 'src_pts' in locals() and 'frame_con_grilla' in locals():
            guardar_snapshot_con_grilla(frame_con_grilla, src_pts)
        else:
            print("[ADVERTENCIA] No se puede guardar la imagen. Verificá que los ArUcos estén detectados.")

cap.release()
cv.destroyAllWindows()
