import socket

HOST = "127.0.0.1"
PORT = 65432

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

print(f"[SERVIDOR] Esperando conexión en {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print(f"[SERVIDOR] Conectado a {addr}")

try:
    while True:
        data = conn.recv(1024)
        if not data:
            break
        print(f"[DATO RECIBIDO] {data.decode()}")
except KeyboardInterrupt:
    print("[SERVIDOR] Finalizado por el usuario.")
finally:
    conn.close()
    server_socket.close()
