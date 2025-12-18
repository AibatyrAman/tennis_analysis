from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance
from utils.device_utils import get_best_device

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        # court_keypoints kullanarak oyuncuları filtreleme işlemi burada yapılacak
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Mesafeleri artan düzende sıralayın
        distances.sort(key=lambda x: x[1])
        # En yakın iki oyuncuyu seçin
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players
        

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            players_dict = self.detect_frame(frame)
            player_detections.append(players_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        device = get_best_device()
        results = self.model.track(frame, persist=True, classes=[0], device=device)[0]
        
        id_name_dict = results.names
        player_dict = {}

        for box in results.boxes:
            # ID kontrolü ekledik (Bazen model ID atayamazsa kod çökmesin diye)
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                
                # Zaten classes=[0] dedik ama yapını bozmamak için bu kontrolü de tuttuk
                object_cls_id = box.cls.tolist()[0]
                object_clc_name = id_name_dict[object_cls_id]

                if object_clc_name == 'person':
                    player_dict[track_id] = result
        
        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # frame üzerine kutuları çiz
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # Kutu rengini ve yazı tipini aynen korudum
                cv2.putText(frame, f"Oyuncu ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            output_video_frames.append(frame)
        
        return output_video_frames