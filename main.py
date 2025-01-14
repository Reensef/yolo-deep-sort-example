import os
import random
import cv2
from ultralytics import YOLO
from tracker import ObjectTracker

# Пути к входному и выходному видеофайлам
INPUT_VIDEO = "in.mp4"
OUTPUT_VIDEO = "out.mp4"

input_path = os.path.join('.', INPUT_VIDEO)
output_path = os.path.join('.', OUTPUT_VIDEO)

# Инициализация захвата видео и записи видео
capture = cv2.VideoCapture(input_path)
success, frame = capture.read()

if not success:
    raise ValueError("Не удалось прочитать видеофайл.")

frame_width = int(frame.shape[1])
frame_height = int(frame.shape[0])
fps = capture.get(cv2.CAP_PROP_FPS)

video_writer = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height)
)

# Инициализация модели YOLO
model = YOLO("yolov8n.pt")

# Инициализация трекера объектов
tracker = ObjectTracker()

# Генерация случайных цветов для отображения ограничивающих рамок
MAX_TRACKS = 10
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(MAX_TRACKS)]

# Порог уверенности для обнаружения объектов
detection_threshold = 0.5

# Обработка кадров видео
while success:
    results = model(frame)

    # Список обнаруженных объектов
    detections = []
    for result in results:
        for box in result.boxes.data.tolist():
            # Извлечение координат рамки, уверенности и ID класса
            x1, y1, x2, y2, score, class_id = [*map(int, box[:4]), box[4], int(box[5])]
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

    # Обновление трекера на основе текущих обнаружений
    tracker.update(frame, detections)

    # Отображение треков на кадре
    for track in tracker.tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        track_id = track.track_id
        color = colors[track_id % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Запись обработанного кадра в выходное видео
    video_writer.write(frame)
    success, frame = capture.read()

# Освобождение ресурсов
capture.release()
video_writer.release()
cv2.destroyAllWindows()