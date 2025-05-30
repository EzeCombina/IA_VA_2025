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
            
            s.settimeout(5)  # Espera hasta 5 segundos por una respuesta
            try:
                respuesta = s.recv(1024)
                print("ESP32 dice:", respuesta.decode())
            except socket.timeout:
                print("No se recibió nada")
            
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nPrograma finalizado.")
#print("Todos los comandos fueron enviados.")

# Armar la página web ... En donde se tenga pulsadores para mapeo, comienzo de ciclos y demas... 
# Comenzar con el código de mapeo deteccion de imágen, escalado, 
# división en celdas y posible edición de la misma 