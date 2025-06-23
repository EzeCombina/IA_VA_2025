import socket
import time

# Configura la IP del ESP32 (por defecto en modo AP es 192.168.4.1)
HOST = '192.168.4.1'
PORT = 3333

comandos = ['Adelante', 'Atras', 'Derecha', 'Izquierda']

print("Conectando al ESP32...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Conectado.")

    #for cmd in comandos:
    #    s.sendall(cmd.encode())
    #    print(f"Enviado: {cmd}")
    #    time.sleep(1)
    try:
        while(1):
            entrada = input("Ingresar dato a mandar al ESP32: ")
            s.sendall(entrada.encode())
            print(f"Enviado: {entrada}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nPrograma finalizado.")
#print("Todos los comandos fueron enviados.")