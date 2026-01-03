from ultralytics import YOLO
import cv2
import sys
import os
import easyocr

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trackers import ScoreboardTracker

def test_scoreboard(input_path, roi=None):
    print(f"--- SKOR TABLOSU TESPİT TESTİ BAŞLIYOR ---")
    print(f"Girdi: {input_path}")

    # ROI (x1, y1, x2, y2) - Default to top left corner if not provided
    if roi is None:
        roi = (100, 100, 600, 300) # Example: Top-leftish area
    print(f"ROI: {roi}")

    try:
        scoreboard_tracker = ScoreboardTracker()
    except Exception as e:
        print(f"Hata: ScoreboardTracker başlatılamadı: {e}")
        return

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    frame_count = 0
    max_frames = 60 # Test first 2 seconds
    
    print("Video işleniyor ve skor okunuyor...")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        
        # Test logic: run process_frame every 10 frames to simulate periodic check
        if frame_count % 10 == 0:
            try:
                # process_frame(frame_num, frame, roi, fps)
                has_changed, current_score = scoreboard_tracker.process_frame(frame_count, frame, roi, fps)
                timestamp = frame_count / fps
                print(f"Frame {frame_count} (@{timestamp:.2f}s): Skor='{current_score}', Değişti={has_changed}")
                
                # Visualize
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Hata frame {frame_count} işlenirken: {e}")

        frame_count += 1
        
    cap.release()
    print("Test tamamlandı.")

if __name__ == "__main__":
    input_file = "input_videos/input_video.mp4"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if os.path.exists(input_file):
        test_scoreboard(input_file)
    else:
        print(f"Uyarı: dosya bulunamadı -> {input_file}")
