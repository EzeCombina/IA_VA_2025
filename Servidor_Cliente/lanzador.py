# lanzar_ambos.py
import subprocess
import time

# Iniciar el servidor en segundo plano
servidor = subprocess.Popen(["python", "servidor.py"])

# Esperar que el servidor esté listo
time.sleep(1)

# Ejecutar el cliente
#subprocess.run(["python", "cliente.py"])

subprocess.run(["python", "conex_esp_servidor.py"])

#subprocess.run(["python", "robot_pos.py"])

# Esperar que el servidor finalice (si querés que se cierre cuando el cliente termina)
servidor.wait()
