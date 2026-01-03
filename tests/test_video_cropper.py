import cv2
import os
import sys
import numpy as np

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.manual_selector import select_corners_manually

def test_video_cropper(input_path):
    print(f"--- VİDEO KIRPMA TESTİ ---")
    print(f"Girdi: {input_path}")

    # Video aç
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Hata: Video açılamadı.")
        return
    
    # İlk kareyi oku
    ret, first_frame = cap.read()
    if not ret:
        print("Hata: Video'dan ilk kare okunamadı.")
        return
    
    # Kullanıcıdan köşe seçmesini iste
    print("Lütfen açılan pencerede kırpmak istediğiniz 4 köşeyi seçin.")
    print("Seçimi tamamlayınca otomatik kapanacaktır.")
    
    corners = select_corners_manually(first_frame)
    if corners is None:
        print("Seçim yapılmadı, işlem iptal ediliyor.")
        return
    
    print(f"Seçilen köşeler: {corners}")
    
    # Kırpma işlemi için Bounding Box hesapla
    # Seçilen 4 noktanın en küçük x, en küçük y, en büyük x, en büyük y değerlerini buluyoruz
    pts = np.array(corners)
    x_min = np.min(pts[:, 0])
    y_min = np.min(pts[:, 1])
    x_max = np.max(pts[:, 0])
    y_max = np.max(pts[:, 1])
    
    # Sınırları kontrol et
    rows, cols, _ = first_frame.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(cols, x_max)
    y_max = min(rows, y_max)
    
    width = x_max - x_min
    height = y_max - y_min
    
    print(f"Kırpılacak Boyut: {width}x{height} (Başlangıç: {x_min},{y_min})")
    
    # Çıktı video hazırlığı
    output_path = f"tests/output_cropped_{os.path.basename(input_path)}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Mac için uygun codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Videoyu baştan sona işle
    # (İlk kareyi zaten okumuştuk ama işaretleyiciler olmasın diye dosyayı yeniden başa sarabiliriz 
    # veya ilk kareyi tekrar okumayız ama frame pointer'ı resetlemek daha temiz)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    print("Video işleniyor...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Kırp
        cropped_frame = frame[y_min:y_max, x_min:x_max]
        
        # Kaydet
        out.write(cropped_frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Kare: {frame_count}")
            
    cap.release()
    out.release()
    print(f"Başarılı! Kırpılmış video kaydedildi: {output_path}")
    

if __name__ == "__main__":
    input_file = "input_videos/input_video.mp4"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if os.path.exists(input_file):
        test_video_cropper(input_file)
    else:
        print(f"Uyarı: dosya bulunamadı -> {input_file}")
