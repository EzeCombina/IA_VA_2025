import socket
import json
import keyboard

HOST = "127.0.0.1"
PORT = 65432

config = {
    "GRID_SIZE": [5, 5],
    "OBSTACLES": [(1, 1), (1, 2), (2, 2), (3, 4), (4, 2)],
    "START": (0, 0),
    "GOAL": (4, 4),
    "ACTIONS": ["UP", "DOWN", "LEFT", "RIGHT"]
}

try:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"[SERVIDOR] Esperando conexión en {HOST}:{PORT}...")
    conn, addr = server_socket.accept()
    print(f"[SERVIDOR] Conectado a {addr}")

    while True:
        print("Presioná la tecla 'e' para enviar la configuración...")
        keyboard.wait("e")
        conn.send(json.dumps(config).encode())
        print("[SERVIDOR] Configuración enviada.")

        data = conn.recv(4096)
        if data:
            print(f"[DATO RECIBIDO] {data.decode()}")
        else:
            print("[SERVIDOR] Cliente desconectado.")
            break

except KeyboardInterrupt:
    print("[SERVIDOR] Finalizado por el usuario.")
finally:
    conn.close()
    server_socket.close()

