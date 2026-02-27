import cv2
import numpy as np
import matplotlib.pyplot as plt

# Inicializar cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

plt.ion()  # Modo interactivo para actualizar histograma
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo recibir el frame")
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convertir imagen a vector 1D (muestra estadística)
    valores = gray.flatten()

    # Cálculos estadísticos
    media = np.mean(valores)
    varianza = np.var(valores)
    desviacion = np.std(valores)

    # Histograma (distribución empírica)
    hist, bins = np.histogram(valores, bins=256, range=(0, 256))
    probabilidades = hist / len(valores)

    # Probabilidad de intensidad mayor a 200
    prob_mayor_200 = np.sum(probabilidades[200:])

    # Mostrar estadísticas en pantalla
    cv2.putText(gray, f"Media: {media:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(gray, f"Varianza: {varianza:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(gray, f"Desv. Std: {desviacion:.2f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(gray, f"P(X > 200): {prob_mayor_200:.4f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Analisis Estadistico en Tiempo Real", gray)

    # Actualizar histograma cada frame
    ax.clear()
    ax.bar(range(256), probabilidades)
    ax.set_title("Distribucion de Probabilidad de Intensidades")
    ax.set_xlabel("Intensidad (0-255)")
    ax.set_ylabel("Probabilidad")
    plt.pause(0.001)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
