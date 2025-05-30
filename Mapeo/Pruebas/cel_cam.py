import cv2

url = "http://192.168.1.6:8080/video"  # Dirección IP del celular
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video desde el celular", frame)
    if cv2.waitKey(1) == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()