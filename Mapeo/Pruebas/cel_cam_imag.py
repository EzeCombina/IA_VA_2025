import cv2 as cv
import numpy as np

# Diccionario ArUco y detector
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

# Cargar imagen a insertar
img_to_insert = cv.imread("aula.png")
if img_to_insert is None:
    raise Exception("No se encontró 'aula.png'")

h_img, w_img = img_to_insert.shape[:2]

# IDs esperados y posiciones
aruco_positions = {
    25: ('TL', 2),  # esquina 2
    33: ('TR', 3),  # esquina 3
    30: ('BR', 0),  # esquina 0
    23: ('BL', 1)   # esquina 1
}

# Abrir cámara IP
cap = cv.VideoCapture("http://192.168.1.52:8080/video")  # ajustá la IP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.flatten()
        detected = {int(id): corner for id, corner in zip(ids, corners)}

        # Verificar si están todos los ArUcos necesarios
        if all(id in detected for id in aruco_positions):
            src_pts = []

            # Obtener puntos desde las esquinas correctas
            for id in [25, 33, 23, 30]:  # orden: TL, TR, BL, BR
                _, corner_index = aruco_positions[id]
                pt = detected[id][0][corner_index]
                src_pts.append(pt)

                # Dibujar ID y puntos para verificar
                cv.polylines(frame, [np.int32(detected[id])], True, (0, 255, 0), 2)
                for i, c in enumerate(detected[id][0]):
                    cv.circle(frame, tuple(c.astype(int)), 4, (255, 0, 0), -1)
                    cv.putText(frame, f"{i}", tuple(c.astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"ID {id}", tuple(detected[id][0][corner_index].astype(int)), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            src_pts = np.array(src_pts, dtype=np.float32)
            dst_pts = np.array([[0, 0], [w_img, 0], [0, h_img], [w_img, h_img]], dtype=np.float32)

            # Homografía: de la imagen al plano detectado
            matrix = cv.getPerspectiveTransform(dst_pts, src_pts)

            # Aplicar homografía para insertar imagen en el frame
            warped = cv.warpPerspective(img_to_insert, matrix, (frame.shape[1], frame.shape[0]))

            # Crear máscara de la imagen insertada
            mask = cv.warpPerspective(np.ones_like(img_to_insert, dtype=np.uint8) * 255, matrix, (frame.shape[1], frame.shape[0]))
            mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            _, mask_binary = cv.threshold(mask_gray, 1, 255, cv.THRESH_BINARY)
            mask_inv = cv.bitwise_not(mask_binary)

            # Combinar imagen insertada con el fondo
            frame_bg = cv.bitwise_and(frame, frame, mask=mask_inv)
            warped_fg = cv.bitwise_and(warped, warped, mask=mask_binary)
            frame = cv.add(frame_bg, warped_fg)

    cv.imshow("Aruco Projection", frame)
    if cv.waitKey(1) == 27:  # ESC para salir
        break

cap.release()
cv.destroyAllWindows()



