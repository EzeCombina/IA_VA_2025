import socket
import time

# === Socket para ESP32 ===
ESP32_IP = '192.168.4.1'
ESP32_PORT = 3333

# === Socket para comunicación local con otro script en la PC ===
LOCAL_SERVER_IP = '127.0.0.1'
LOCAL_SERVER_PORT = 65432

print("Conectando al ESP32...")
esp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
esp_socket.connect((ESP32_IP, ESP32_PORT))
print("Conectado al ESP32.")

print("Conectando al servidor local...")
local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    local_socket.connect((LOCAL_SERVER_IP, LOCAL_SERVER_PORT))
    print("Conectado al servidor local.")
except ConnectionRefusedError:
    print("No se pudo conectar al servidor local.")
    local_socket = None

try:
    while True:
        entrada = input("Ingresar dato a mandar al ESP32: ")

        # Enviar al ESP32
        esp_socket.sendall(entrada.encode())
        print(f"[ESP32] Enviado: {entrada}")
        esp_socket.settimeout(5)
        try:
            respuesta = esp_socket.recv(1024)
            print("[ESP32] Respuesta:", respuesta.decode())
        except socket.timeout:
            print("[ESP32] No se recibió respuesta")

        # Enviar también al servidor local (opcional)
        #if local_socket:
        #    try:
        #        local_socket.sendall(entrada.encode())
        #        print(f"[LOCAL] Enviado: {entrada}")
        #    except:
        #        print("[LOCAL] Error al enviar al servidor local.")

        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    print("\nCerrando conexiones...")
    esp_socket.close()
    if local_socket:
        local_socket.close()
    print("Programa finalizado.")

