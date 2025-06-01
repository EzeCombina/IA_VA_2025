# lanzar_ambos.py
import subprocess
import time

# Iniciar el servidor en segundo plano
servidor = subprocess.Popen(["python", "servidor_q_learning.py"])

# Esperar que el servidor esté listo
time.sleep(1)

# Ejecutar el cliente
#subprocess.run(["python", "cliente.py"])

subprocess.run(["python", "q_learning.py"])

#subprocess.run(["python", "robot_pos.py"])

# Esperar que el servidor finalice (si querés que se cierre cuando el cliente termina)
servidor.wait()
