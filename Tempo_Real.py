from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Carrega modelo
model = YOLO("pesos/best.pt")

# Abre câmera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Erro ao abrir câmera")
    exit()

plt.ion()  # modo interativo
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO
    results = model(frame)

    for r in results:
        frame = r.plot()

    # Converte BGR -> RGB (matplotlib usa RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ax.clear()
    ax.imshow(frame_rgb)
    ax.set_title("YOLO Tempo Real")
    ax.axis("off")

    plt.pause(0.001)

cap.release()
plt.close()