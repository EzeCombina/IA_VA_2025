from flask import Flask, render_template, Response, request, jsonify
import cv2

app = Flask(__name__)

GRID_SIZE = 40

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index0.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/guardar_config', methods=['POST'])
def guardar_config():
    data = request.json
    puntos = data.get("puntos", {})

    # Procesar puntos finales (pueden venir como lista o undefined)
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
