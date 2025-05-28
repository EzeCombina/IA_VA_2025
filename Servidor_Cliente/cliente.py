# cliente.py
import socket

HOST = 'localhost'
PUERTO = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PUERTO))
    mensaje = input("Escribí un mensaje para enviar al servidor: ")
    s.sendall(mensaje.encode())
    print("Mensaje enviado.")
