from ultralytics import YOLO
import cv2
import os
from collections import Counter

# Carrega modelo
model = YOLO("pesos/best.pt")

# Caminho da imagem
image_path = "img/eu.png"

# Pasta de saída
output_dir = "resultados"
os.makedirs(output_dir, exist_ok=True)

# Extrai nome original (sem extensão)
file_name = os.path.basename(image_path)
base_name, original_ext = os.path.splitext(file_name)

# Extensão de saída da imagem
ext = ".jpg"

# Gera nome incremental baseado no original
i = 1
while True:
    output_img = os.path.join(output_dir, f"{base_name}({i}){ext}")
    output_txt = os.path.join(output_dir, f"{base_name}({i}).txt")
    if not os.path.exists(output_img) and not os.path.exists(output_txt):
        break
    i += 1

# Predição
results = model(image_path)

for r in results:
    img = r.plot()

    # salva imagem
    cv2.imwrite(output_img, img)

    # pega classes detectadas
    class_ids = r.boxes.cls.tolist()
    names = model.names

    # conta ocorrências
    counts = Counter([names[int(c)] for c in class_ids])

    # salva txt
    with open(output_txt, "w", encoding="utf-8") as f:
        for cls, qtd in counts.items():
            f.write(f"{cls}: {qtd}\n")

print(f"Imagem salva em: {output_img}")
print(f"TXT salvo em: {output_txt}")