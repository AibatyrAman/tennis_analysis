import cv2
import os
import sys

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trackers import PoseTracker

def test_pose(input_path, model_path="models/yolo11x-pose.pt"):
    print(f"--- İSKELET TESPİT TESTİ BAŞLIYOR ---")
    print(f"Girdi: {input_path}")

    pose_tracker = PoseTracker(model_path=model_path)

    cap = cv2.VideoCapture(input_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # İlk 20 kareyi test et
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
        print("Hata: Video okunamadı.")
        return

    print("İskeletler tespit ediliyor...")
    pose_detections = pose_tracker.detect_frames(frames, read_from_stub=False)

    print("Çizim yapılıyor...")
    output_frames = pose_tracker.draw_bboxes(frames, pose_detections)

    output_path = f"tests/output_pose_{os.path.basename(input_path)}"
    height, width, _ = output_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in output_frames:
        out.write(frame)
    out.release()
    print(f"Başarılı! İskelet videosu kaydedildi: {output_path}")

if __name__ == "__main__":
    input_file = "input_videos/input_video.mp4"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if os.path.exists(input_file):
        test_pose(input_file)
    else:
        print(f"Uyarı: dosya bulunamadı -> {input_file}")
