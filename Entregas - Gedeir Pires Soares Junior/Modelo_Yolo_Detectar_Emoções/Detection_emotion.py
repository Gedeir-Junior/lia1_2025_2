import cv2
import time
import psutil
import pandas as pd
from ultralytics import YOLO

model = YOLO('model/best_2_Emotions.pt')
video = cv2.VideoCapture(0)

cores = {
    "Happy": (0, 255, 255),       # amarelo
    "Sad": (255, 0, 0),      # azul
    "Surprise": (0, 255, 255),    # verde
    "Neutral": (255, 255, 0),    # ciano
    "Anger": (0, 0, 255),     # vermelho
    "Fear": (255, 0, 255),      # rosa
    "Contempt": (0,0,0),  #preto
    "Disgust": (0,125,255),    #laranja

}

if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

output_video = cv2.VideoWriter(
    'video/saved_predictions.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (frame_width, frame_height)
)

metricas = []

while True:
    check, img = video.read()
    if not check:
        print("Não foi possível ler o frame. Finalizando...")
        break

    inicio = time.time()

    #resized = cv2.resize(img, (640,640))

    # Predição YOLO
    results = model(img, verbose=False)[0]
    nomes = results.names

    for box in results.boxes:
        # Coordenadas da bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        w, h = x2 - x1, y2 - y1
        area = w * h
        frame_area = frame_width * frame_height
        aspect_ratio = w / h if h > 0 else 0

        # Classe
        cls = int(box.cls.item())
        nomeClasse = nomes[cls]
        conf = float(box.conf.item())

        # Confiança
        conf = float(box.conf.item())


        if 0.5 > conf:
            continue

        #if area < 0.01 * frame_area:  # < 1% do frame (muito pequeno)
            #continue
        #if area > 0.4 * frame_area:  # > 40% do frame (muito grande)
            #continue

        # Filtrar por proporção (rostos geralmente ~0.8 a 1.5)
        #if not (0.6 < aspect_ratio < 1.6):
            #continue

        # Texto na imagem

        # Pega a cor da classe ou branco se não definida
        cor = cores.get(nomeClasse, (255, 255, 255))

        # Desenha bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)


        texto = f'{nomeClasse} - {conf:.2f}'

        cv2.putText(img, texto, (x1, y1+25),
                    cv2.FONT_HERSHEY_COMPLEX, 1, cor, 2)
        

    # Tempo de inferência e métricas
    tempo_inferencia = time.time() - inicio
    fps_atual = 2 / tempo_inferencia if tempo_inferencia > 0 else 0
    uso_cpu = psutil.cpu_percent()
    uso_memoria = psutil.virtual_memory().percent

    metricas.append({
        "Tempo_inferencia (s)": round(tempo_inferencia, 4),
        "FPS": round(fps_atual, 2),
        "Uso_CPU (%)": uso_cpu,
        "Uso_Memória (%)": uso_memoria
    })

    # Exibe e grava frame
    cv2.imshow('IMG', img)
    output_video.write(img)

    if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
        break

video.release()
output_video.release()
cv2.destroyAllWindows()

df = pd.DataFrame(metricas)
df.to_excel("video/metricas_yolo.xlsx", index=False)
print("Métricas salvas em 'video/metricas_yolo.xlsx'")