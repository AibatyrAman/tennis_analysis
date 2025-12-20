import cv2
import os
import sys

# Proje ana dizinini path'e ekle (modülleri bulabilmek için)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from court_line_detector import CourtLineDetector

def test_court_detection(input_path, model_path="models/keypoints_model_50.pth"):
    print(f"--- KORT TESPİT TESTİ BAŞLIYOR ---")
    print(f"Girdi: {input_path}")
    print(f"Model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı -> {model_path}")
        return

    detector = CourtLineDetector(model_path)
    
    ext = os.path.splitext(input_path)[1].lower()
    output_path = f"tests/output_court_{os.path.basename(input_path)}"

    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image = cv2.imread(input_path)
        if image is None:
            print("Hata: Resim okunamadı.")
            return

        keypoints = detector.predict(image)
        output_image = detector.draw_keypoints(image, keypoints)
        cv2.imwrite(output_path, output_image)
        print(f"Başarılı! Çıktı kaydedildi: {output_path}")

    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        cap = cv2.VideoCapture(input_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Test için sadece ilk 30 kareyi (1 saniye) işleyelim
        max_frames = 22800
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret or count >= max_frames:
                break
            frames.append(frame)
            count += 1
        cap.release()

        print(f"{len(frames)} kare işleniyor...")
        output_frames = detector.draw_keypoints_on_video(frames)
        
        if output_frames:
            height, width, _ = output_frames[0].shape
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            for frame in output_frames:
                out.write(frame)
            out.release()
            print(f"Başarılı! Test videosu kaydedildi: {output_path}")

if __name__ == "__main__":
    # Varsayılan test dosyası (kullanıcı değiştirebilir)
    input_file = "input_videos/input_video.mp4" 
    
    # Komut satırından dosya yolunu alabiliriz
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if os.path.exists(input_file):
        test_court_detection(input_file)
    else:
        print(f"Uyarı: Test dosyası bulunamadı ({input_file}). Lütfen geçerli bir yol belirtin.")
        print("Örnek: python tests/test_court.py input_videos/mac.mp4")
