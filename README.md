# IA_VA_2025

# Comprar Afiches (Blancos y Negros) 
# Imprimir tableros de ajedrez (4 o 5) (No necesario)
# Imprimir ArUco (Listo)
# Ver manera de colocar la cámara 
# Traer cinta 
# Probar a baja escala el ArUco y el dominio, junto con la calibración y la determinación de la pocisión (Listo)
# Agregar gráfico de la trayectória 
# Hacer un programa provisorio que detecte la posición del robot, se ingrese una posición destino y el robot vaya a la misma. 

# Hacer varios scripts en donde: El primero → Haga toda el procesamiento del Aprendizaje por Refuerzo y envie el comando de movimiento al ESP32
#                                           → Antes de mandar el siguiente movimiento comprobar si ya termino el anterior
#                                           → Antes del siguiente movimiento corroborar si esta en la posicion correcta comparando 
#                                             la posicion predecida con la real. 
#                                El segundo → Cree la matriz con los ArUcos, ubique al robot y pase su ubicación al script 1 cuando se lo solicite
#                                El tercero → Servidor web. Con el mismo se va a poder seleccionar punto inicial, final, y ostáculos en la matriz
#                                             generada en el segundo script. Además de darle inicio al programa. 