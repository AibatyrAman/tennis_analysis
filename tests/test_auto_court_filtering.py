import cv2
import os
import sys
import numpy as np

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from court_line_detector import CourtLineDetector

def test_auto_court_filtering(input_path, model_path="models/keypoints_model_50.pt"):
    print(f"--- OTOMATİK AKSİYON FİLTRELEME TESTİ (KORT BAZLI) ---")
    print(f"Girdi Video: {input_path}")
    print(f"Model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Hata: Model bulunamadı -> {model_path}")
        return

    # Detector başlat
    detector = CourtLineDetector(model_path)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Hata: Video açılamadı.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Toplam Kare Sayısı: {total_frames}")

    # Çıktı videosu için hazırlık
    output_path = f"tests/output_filtered_{os.path.basename(input_path)}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Çıktı dosyası: {output_path}")
    print("Kareler analiz ediliyor ve filtreleniyor...")
    
    kept_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Tahmin yap
        keypoints = detector.predict(frame)
        
        # 0, 1, 2, 3 Nolu Noktalar (Kort Köşeleri)
        # Genelde: 0,1 -> Üst (Uzak) Köşeler / 2,3 -> Alt (Yakın) Köşeler
        p0 = keypoints[0:2] # x,y
        p1 = keypoints[2:4]
        p2 = keypoints[4:6]
        p3 = keypoints[6:8]
        
        # Basit "Valid Court" (Geçerli Kort) Kontrolü
        # 1. Noktaların hepsi frame sınırları içinde mi? (Esneme payı bırakabiliriz)
        margin = -100 # Model bazen dışarı taşabilir, o yüzden biraz pay verelim
        
        def is_in_bounds(pt):
            x, y = pt
            return (margin < x < width - margin) and (margin < y < height - margin)
        
        valid_corners = all([is_in_bounds(p0), is_in_bounds(p1), is_in_bounds(p2), is_in_bounds(p3)])
        
        # 2. Üst kenar genişliği ve Alt kenar genişliği kontrolü
        # Bir tenis kortu perspektifte yamuk (trapezoid) görünür.
        # Alt kenar (2-3), üst kenardan (0-1) daha geniş olmalı.
        dist_top = np.linalg.norm(p0 - p1)
        dist_bottom = np.linalg.norm(p2 - p3)
        
        # Eşik değerler (Video çözünürlüğüne göre değişebilir ama oran olarak bakabiliriz)
        # Alt kenar, üst kenarın en az 2 katı kadar olmalı (tahmini) ve belli bir pikselden büyük olmalı
        geometric_validity = (dist_bottom > dist_top) and (dist_top > 50) and (dist_bottom > 200)

        if valid_corners and geometric_validity:
            # Görselleştirme (opsiyonel, test çıktısında görmek için)
            debug_frame = frame.copy()
            # Köşeleri çiz
            pts = [p0, p1, p3, p2] # Çizim sırası polygon için
            for pt in pts:
                cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)
            
            # Doğrudan diske yaz, listeye ekleme
            out.write(debug_frame)
            kept_count += 1
            status = "SAKLANDI"
        else:
            status = "ATILDI"

        if frame_idx % 50 == 0:
            print(f"Kare {frame_idx}: {status} | TopDist: {dist_top:.1f}, BotDist: {dist_bottom:.1f}")
            
        frame_idx += 1

    cap.release()
    out.release()
    
    print(f"Filtreleme Tamamlandı. {kept_count} / {total_frames} kare saklandı.")

if __name__ == "__main__":
    input_file = "input_videos/input_video.mp4"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if os.path.exists(input_file):
        test_auto_court_filtering(input_file)
    else:
        print(f"Dosya bulunamadı: {input_file}")
