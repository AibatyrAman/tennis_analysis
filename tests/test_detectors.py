import cv2
import os
import sys

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trackers import PlayerTracker, BallTracker

def test_detectors(input_path, model_yolo="models/yolov8x.pt", model_ball="models/yeni_model.pt"):
    print(f"--- OYUNCU VE TOP TESPİT TESTİ BAŞLIYOR ---")
    print(f"Girdi: {input_path}")

    # Initialize Trackers
    player_tracker = PlayerTracker(model_path=model_yolo)
    ball_tracker = BallTracker(model_path=model_ball)

    cap = cv2.VideoCapture(input_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Test için 3. saniyeden itibaren 20 kare alalım (aksiyonun olduğu yerler genelde ortadadır)
    start_frame = int(fps * 3)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    max_frames = 20
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        frames.append(frame)
        count += 1
    cap.release()
    
    if not frames:
        print("Hata: Video'dan kare okunamadı.")
        return

    print(f"{len(frames)} kare üzerinde tespit yapılıyor...")

    # Detect
    print("- Oyuncular tespit ediliyor...")
    # read_from_stub=False ile her seferinde taze tespit yapıyoruz
    player_detections = player_tracker.detect_frames(frames, read_from_stub=False)
    
    print("- Top tespit ediliyor...")
    ball_detections = ball_tracker.detect_frames(frames, read_from_stub=False)
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Draw
    output_frames = player_tracker.draw_bboxes(frames, player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)

    # Save
    output_path = f"tests/output_detectors_{os.path.basename(input_path)}"
    height, width, _ = output_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in output_frames:
        out.write(frame)
    out.release()
    print(f"Başarılı! Tespit videosu kaydedildi: {output_path}")

if __name__ == "__main__":
    input_file = "input_videos/input_video.mp4"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if os.path.exists(input_file):
        test_detectors(input_file)
    else:
        print(f"Uyarı: Test dosyası bulunamadı ({input_file}).")
