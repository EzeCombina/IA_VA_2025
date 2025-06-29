from flask import Flask, render_template, Response, request, jsonify
import cv2 as cv
import numpy as np
import time
import datetime
import socket

app = Flask(__name__)

# === CONFIGURACIÓN TCP ===
TCP_IP = "192.168.4.1"
TCP_PORT = 3333

def esperar_conexion_esp32():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TCP_IP, TCP_PORT))
            print("[MAPEO] Conectado al ESP32.")
            return s
        except (ConnectionRefusedError, OSError):
            print("[MAPEO] ESP32 ocupado. Reintentando en 1 segundos...")
            time.sleep(1)

sock = esperar_conexion_esp32()

GRID_SIZE = 40
camera = cv.VideoCapture(0)

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

aruco_positions = {
    25: ('TL', 2),
    33: ('TR', 3),
    30: ('BR', 0),
    23: ('BL', 1)
}
mobile_id = 99

# Grilla
grid_width = 400
grid_height = 400
step = GRID_SIZE

last_sent_cell = None
last_sent_time = 0
send_interval = 1.0

dst_pts = np.array([
    [0, 0],
    [grid_width, 0],
    [grid_width, grid_height],
    [0, grid_height]
], dtype=np.float32)

@app.route('/')
def index():
    return render_template('index0.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global last_sent_cell, last_sent_time
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            ids = ids.flatten()
            detected = {int(id): corner for id, corner in zip(ids, corners)}

            if all(id in detected for id in aruco_positions):
                src_pts = []
                for id in [25, 33, 30, 23]:
                    _, idx = aruco_positions[id]
                    pt = detected[id][0][idx]
                    src_pts.append(pt)
                src_pts = np.array(src_pts, dtype=np.float32)

                h_matrix = cv.getPerspectiveTransform(dst_pts, src_pts)
                inv_h_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)

                grid_img = np.zeros((grid_height, grid_width, 4), dtype=np.uint8)
                line_color = (255, 0, 0, 255)

                for x in range(0, grid_width, step):
                    cv.line(grid_img, (x, 0), (x, grid_height), line_color, 1)
                for y in range(0, grid_height, step):
                    cv.line(grid_img, (0, y), (grid_width, y), line_color, 1)

                warped = cv.warpPerspective(grid_img, h_matrix, (frame.shape[1], frame.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
                warped_bgr = warped[..., :3]
                alpha_mask = warped[..., 3]
                alpha_mask_3 = cv.merge([alpha_mask]*3)
                alpha = alpha_mask_3.astype(float) / 255.0
                frame_float = frame.astype(float)
                warped_float = warped_bgr.astype(float)
                frame = cv.convertScaleAbs(warped_float * alpha + frame_float * (1 - alpha))

                if mobile_id in detected:
                    mobile_corners = detected[mobile_id][0]
                    mobile_corners = mobile_corners.reshape(-1, 1, 2)
                    transformed_corners = cv.perspectiveTransform(mobile_corners, inv_h_matrix)
                    xs = transformed_corners[:, 0, 0]
                    ys = transformed_corners[:, 0, 1]
                    cols = xs // step
                    rows = ys // step

                    if np.all(cols == cols[0]) and np.all(rows == rows[0]):
                        col = int(cols[0])
                        row = int(rows[0])

                        if 0 <= col < (grid_width // step) and 0 <= row < (grid_height // step):
                            x_center = np.mean(xs)
                            y_center = np.mean(ys)
                            cell_x_min = col * step
                            cell_x_max = (col + 1) * step
                            cell_y_min = row * step
                            cell_y_max = (row + 1) * step
                            margin_ratio = 0.2
                            margin_x = step * margin_ratio
                            margin_y = step * margin_ratio
                            central_x_min = cell_x_min + margin_x
                            central_x_max = cell_x_max - margin_x
                            central_y_min = cell_y_min + margin_y
                            central_y_max = cell_y_max - margin_y

                            if central_x_min <= x_center <= central_x_max and central_y_min <= y_center <= central_y_max:
                                mensaje = f"POS {row},{col}\n"
                                current_time = time.time()
                                if (last_sent_cell != (row, col)) or (current_time - last_sent_time > send_interval):
                                    print(f"[TCP] Enviando: {mensaje.strip()}")
                                    try:
                                        if sock:
                                            sock.send(mensaje.encode())
                                    except Exception as e:
                                        print(f"[TCP] Error al enviar: {e}")
                                    last_sent_cell = (row, col)
                                    last_sent_time = current_time

        ret, buffer = cv.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/guardar_config', methods=['POST'])
def guardar_config():
    data = request.json
    puntos = data.get("puntos", {})
    finales = puntos.get("finales", [])
    goals = [(int(p["x"] // GRID_SIZE), int(p["y"] // GRID_SIZE)) for p in finales]
    config = {
        "GRID_SIZE": GRID_SIZE,
        "OBSTACLES": [(int(o['x'] // GRID_SIZE), int(o['y'] // GRID_SIZE)) for o in puntos.get("obstaculos", [])],
        "START": (int(puntos["I"]["x"] // GRID_SIZE), int(puntos["I"]["y"] // GRID_SIZE)) if "I" in puntos else (0, 0),
        "GOALS": goals,
        "ACTIONS": ["UP", "DOWN", "LEFT", "RIGHT"]
    }
    try:
        with open("configuracion.txt", "w") as f:
            f.write(str(config))
        return jsonify({"success": True, "message": "Archivo guardado correctamente."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
