from ultralytics import YOLO
import cv2
import pickle
import sys
import os
import numpy as np

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trackers import PlayerTracker
from court_line_detector import CourtLineDetector

def test_player_recovery(input_path, model_path="models/yolov8x.pt", court_model_path="models/keypoints_model_50.pt"):
    print(f"--- OYUNCU TAKİP & KURTARMA TESTİ BAŞLIYOR ---")
    print(f"Girdi: {input_path}")
    
    # 1. Initialize Detectors
    player_tracker = PlayerTracker(model_path=model_path)
    court_line_detector = CourtLineDetector(court_model_path)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Video açılamadı!")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # OUTPUT VIDEO
    output_path = f"tests/output_recovery_{os.path.basename(input_path)}"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frames = []
    MAX_FRAMES = 22000 # Test first 200 frames (enough to see tracking loss usually)
    
    print(f"İlk {MAX_FRAMES} kare okunuyor...")
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= MAX_FRAMES:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("Hiç kare okunamadı.")
        return

    # 2. Detect Court Lines (Once for the first frame)
    print("Kort çizgileri tespit ediliyor...")
    court_keypoints = court_line_detector.predict(frames[0])
    # Note: court_line_detector.predict usually expects an image and returns keypoints.
    # We'll assume the standard usage:
    # If predict returns a list of outputs, take the first one.
   
    # 3. Detect Players (Batch or per frame)
    print("Oyuncular tespit ediliyor (YOLO)...")
    raw_player_detections = player_tracker.detect_frames(frames, read_from_stub=False)
    
    # 4. Filter and Track
    print("Oyuncular filtreleniyor ve takip ediliyor...")
    # NOTE: We can simulate 'manual corners' if we want, but let's test automatic first
    filtered_player_detections = player_tracker.choose_and_filter_players(court_keypoints, raw_player_detections, corners=None)
    
    # 5. Draw and Save
    print("Video oluşturuluyor...")
    for i, frame in enumerate(frames):
        # Draw all raw detections in GRAY for debugging
        raw_dict = raw_player_detections[i]
        for track_id, bbox in raw_dict.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 100), 1)
            cv2.putText(frame, f"{track_id}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
            
        # Draw Filtered detections in GREEN/RED
        filtered_dict = filtered_player_detections[i]
        for track_id, bbox in filtered_dict.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"Player {track_id}", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Info
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracked: {list(filtered_dict.keys())}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(frame)
        
    out.release()
    print(f"Test tamamlandı: {output_path}")

if __name__ == "__main__":
    input_file = "input_videos/input_video.mp4"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if os.path.exists(input_file):
        test_player_recovery(input_file)
    else:
        print(f"Dosya bulunamadı: {input_file} (Default kullanarak devam ediliyor...)")
        if os.path.exists("input_videos/input_video.mp4"):
             test_player_recovery("input_videos/input_video.mp4")
