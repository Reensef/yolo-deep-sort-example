import numpy as np
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching

class ObjectTracker:
    def __init__(self, model_path='model_data/mars-small128.pb', max_cosine_distance=0.4, nn_budget=None):
        # Создание энкодера для извлечения характеристик объектов. Используется предобученная модель.
        self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
        
        # Создание метрики для сравнения объектов с использованием косинусного расстояния.
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        # Инициализация трекера с указанной метрикой.
        self.tracker = DeepSortTracker(metric)
        
        # Список активных треков (объектов, которые отслеживаются).
        self.tracks = []

    def update(self, frame, detections):
        # Если нет обнаружений, обновляем трекер пустым списком.
        if not detections:
            self._predict_and_update([])
            return

        # Преобразование координат ограничивающих рамок в формат [x, y, width, height].
        bboxes = np.array([det[:4] for det in detections])
        bboxes[:, 2:] -= bboxes[:, :2]
        
        # Извлечение оценок для каждого обнаружения.
        scores = np.array([det[4] for det in detections])

        # Генерация признаков для каждого обнаружения с использованием энкодера.
        features = self.encoder(frame, bboxes)

        # Создание объектов Detection для трекера.
        detection_objects = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]
        
        # Обновление трекера с использованием текущих обнаружений.
        self._predict_and_update(detection_objects)

    def _predict_and_update(self, detection_objects):
        # Предсказание новых позиций треков на основе модели движения.
        self.tracker.predict()
        
        # Обновление трекера на основе новых обнаружений.
        self.tracker.update(detection_objects)
        
        # Обновление списка активных треков.
        self._refresh_tracks()

    def _refresh_tracks(self):
        # Обновление списка треков, которые подтверждены и недавно обновлены.
        self.tracks = [
            Track(track.track_id, track.to_tlbr())
            for track in self.tracker.tracks
            if track.is_confirmed() and track.time_since_update <= 1
        ]

class Track:
    def __init__(self, track_id, bbox):
        # Уникальный идентификатор трека.
        self.track_id = track_id
        
        # Координаты ограничивающей рамки объекта в формате [x1, y1, x2, y2].
        self.bbox = bbox
