import cv2 as cv
import numpy as np

# Dirección de la cámara IP
ip_url = "http://192.168.1.6:8080/video"  # <-- Cambiá esto por tu dirección real

# Cargamos el diccionario de ArUco
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Creamos los parámetros de detección
parameters = cv.aruco.DetectorParameters()

# Creamos el detector
detector = cv.aruco.ArucoDetector(dictionary, parameters)

# Conectamos a la cámara IP
cap = cv.VideoCapture(ip_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    # Detección de marcadores ArUco
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        print("\nesquinas: {}".format(corners))
        print("ids: {}".format(ids))
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

    # Mostrar la imagen
    cv.imshow("ArUco IP Cam", frame)
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()

