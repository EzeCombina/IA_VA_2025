import cv2 as cv
import numpy as np
import time

# Diccionario y detector ArUco
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
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

# Colores
GRID_COLOR = (255, 0, 0)
MARKER_COLOR = (0, 255, 255)

# Cámara
cap = cv.VideoCapture("http://192.168.1.6:8080/video")

last_print = 0
print_interval = 3  # segundos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.flatten()
        detected = {int(id): corner for id, corner in zip(ids, corners)}

        if all(id in detected for id in aruco_positions):
            src_pts = []
            # Nuevo orden: TL, TR, BR, BL
            for id in [25, 33, 30, 23]:  # TL, TR, BR, BL
                _, idx = aruco_positions[id]
                pt = detected[id][0][idx]
                src_pts.append(pt)
            src_pts = np.array(src_pts, dtype=np.float32)

            # Coordenadas destino para la grilla (en el mismo orden)
            dst_pts = np.array([
                [0, 0],                  # TL
                [grid_width, 0],         # TR
                [grid_width, grid_height],# BR
                [0, grid_height]         # BL
            ], dtype=np.float32)

            # Homografía para insertar la grilla
            h_matrix = cv.getPerspectiveTransform(dst_pts, src_pts)
            inv_h_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)

            # Crear matriz visual
            grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
            step = 50
            for x in range(0, grid_width, step):
                cv.line(grid_img, (x, 0), (x, grid_height), GRID_COLOR, 1)
            for y in range(0, grid_height, step):
                cv.line(grid_img, (0, y), (grid_width, y), GRID_COLOR, 1)

            # Proyectar grilla
            warped = cv.warpPerspective(grid_img, h_matrix, (frame.shape[1], frame.shape[0]))

            # Máscara y fusión
            mask = np.zeros_like(frame)
            cv.fillConvexPoly(mask, np.int32(src_pts), (255, 255, 255))
            inv_mask = cv.bitwise_not(mask)

            frame_bg = cv.bitwise_and(frame, inv_mask)
            warped_fg = cv.bitwise_and(warped, mask)
            frame = cv.add(frame_bg, warped_fg)

            # Detectar marcador móvil
            if mobile_id in detected:
                mobile_corner = detected[mobile_id][0]
                center = np.mean(mobile_corner, axis=0).reshape(1, 1, 2)

                # Transformar al sistema de la grilla
                transformed = cv.perspectiveTransform(center, inv_h_matrix)
                x, y = transformed[0][0]

                # Dibujar ubicación
                cv.circle(frame, tuple(center[0][0].astype(int)), 8, MARKER_COLOR, -1)
                cv.putText(frame, f"Rel pos: ({x:.1f}, {y:.1f})", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, MARKER_COLOR, 2)

                # Mostrar cada cierto tiempo
                if time.time() - last_print > print_interval:
                    print(f"Posición relativa del marcador {mobile_id}: x={x:.1f}, y={y:.1f}")
                    last_print = time.time()

    cv.imshow("Tracking ArUco", frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()

