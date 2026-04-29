from ultralytics import YOLO
import cv2
import glob
import os
from google.colab.patches import cv2_imshow

model = YOLO("/content/runs/detect/train-3/weights/best.pt")

base_path = "/content/runs/detect"
os.makedirs(base_path, exist_ok=True)

# acha o maior índice existente de predict-N
existing = [d for d in os.listdir(base_path) if d.startswith("predict")]

max_id = 0
for d in existing:
    try:
        num = int(d.split("-")[-1])
        if num > max_id:
            max_id = num
    except:
        pass

next_id = max_id + 1
save_folder = os.path.join(base_path, f"predict-{next_id}")
os.makedirs(save_folder, exist_ok=True)

imagens = glob.glob("/content/datasets/YOLO_BROCAS/images/test/*.jpg")

# cria faixas de 10 em 10
faixas = {f"{i}-{i+10}": 0 for i in range(0, 100, 10)}

# ===== FUNÇÃO NMS MANUAL =====
def filtrar_boxes(boxes, iou_thresh=0.25):
    if len(boxes) == 0:
        return []

    boxes_np = []
    for b in boxes:
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        conf = float(b.conf[0])
        boxes_np.append([x1, y1, x2, y2, conf])

    boxes_np = sorted(boxes_np, key=lambda x: x[4], reverse=True)

    selecionadas = []

    while boxes_np:
        melhor = boxes_np.pop(0)
        selecionadas.append(melhor)

        novas = []
        for box in boxes_np:
            xx1 = max(melhor[0], box[0])
            yy1 = max(melhor[1], box[1])
            xx2 = min(melhor[2], box[2])
            yy2 = min(melhor[3], box[3])

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            area1 = (melhor[2]-melhor[0]) * (melhor[3]-melhor[1])
            area2 = (box[2]-box[0]) * (box[3]-box[1])
            union = area1 + area2 - inter

            iou = inter / union if union > 0 else 0

            if iou < iou_thresh:
                novas.append(box)

        boxes_np = novas

    return selecionadas

for idx, caminho in enumerate(imagens):
    results = model(caminho, conf=0.3, iou=0.5)
    img = cv2.imread(caminho)

    count_50_plus = 0
    count_50_minus = 0

    for r in results:
        boxes_filtradas = filtrar_boxes(r.boxes, iou_thresh=0.25)

        for box in boxes_filtradas:
            x1, y1, x2, y2, conf = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf_percent = conf * 100

            if conf >= 0.5:
                count_50_plus += 1
            else:
                count_50_minus += 1

            faixa_idx = int(conf_percent // 10) * 10
            if faixa_idx >= 100:
                faixa_idx = 90
            faixa_key = f"{faixa_idx}-{faixa_idx+10}"
            faixas[faixa_key] += 1

            if conf < 0.10:
                color = (0, 0, 255)
            elif conf < 0.50:
                color = (0, 165, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{conf_percent:.1f}%"
            cv2.putText(img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # espaço no topo
    top_space = 40
    img = cv2.copyMakeBorder(img, top_space, 0, 0, 0,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # cores
    color_plus = (255, 0, 0)
    color_minus = (0, 165, 255)

    cv2.putText(img, f"50%+ = {count_50_plus}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_plus, 2)

    cv2.putText(img, f"50%- = {count_50_minus}", (250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_minus, 2)

    cv2_imshow(img)
    cv2.imwrite(os.path.join(save_folder, f"img_{idx}.jpg"), img)

# ===== RESUMO =====
print("===== RESUMO PREDIÇÃO =====")

total = sum(faixas.values())

for faixa, valor in faixas.items():
    if total > 0:
        porcentagem = (valor / total) * 100
        print(f"{faixa}%: {valor}   | {porcentagem:.1f}%")
    else:
        print(f"{faixa}%: {valor}   | 0%")

print("Salvo em:", save_folder)