from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

DATA_FILE = 'puntos.json' # Archivo para guardar los puntos del agente
CONFIG_FILE = 'config.txt' # Archivo de configuración para el agente

# Cargar puntos guardados si existen
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as f:
        puntos_guardados = json.load(f)
else:
    puntos_guardados = {}

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/datos')
def obtener_datos():
    return jsonify(puntos_guardados)

@app.route('/guardar_puntos', methods=['POST'])
def guardar_puntos():
    global puntos_guardados
    puntos_guardados = request.json
    with open(DATA_FILE, 'w') as f:
        json.dump(puntos_guardados, f)
    return '', 204

@app.route('/enviar_config', methods=['POST'])
def enviar_config():
    try:
        grid_size = [20, 15]  # Tamaño del grid
        puntos = puntos_guardados.copy()

        # Obtener puntos finales
        finales = [k for k in puntos if k.startswith("F")]
        goals = [
            (int(puntos[k]["x"] // 40), int(puntos[k]["y"] // 40))
            for k in sorted(finales)
        ]

        # Crear el diccionario de configuración
        config = {
            "GRID_SIZE": grid_size,
            "OBSTACLES": [(int(o['x'] // 40), int(o['y'] // 40)) for o in puntos.get("obstaculos", [])],
            "START": (int(puntos["I"]["x"] // 40), int(puntos["I"]["y"] // 40)) if "I" in puntos else (0, 0),
            "GOALS": goals,
            "ACTIONS": ["UP", "DOWN", "LEFT", "RIGHT"]
        }

        # Guardar en archivo de texto como string con formato de Python
        with open(CONFIG_FILE, 'w') as f:
            f.write("config = {\n")
            f.write(f'    "GRID_SIZE": {config["GRID_SIZE"]},\n')
            f.write(f'    "OBSTACLES": {config["OBSTACLES"]},\n')
            f.write(f'    "START": {config["START"]},\n')
            f.write(f'    "GOALS": {config["GOALS"]},\n')
            f.write(f'    "ACTIONS": {config["ACTIONS"]}\n')
            f.write("}")

        return jsonify({"status": "ok", "mensaje": "Configuración guardada en config.txt"}), 200

    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
