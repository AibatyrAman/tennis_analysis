import sys
import os

# Add current directory to path if running directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))

from tennis_detector import TennisCourtDetector
import cv2
import numpy as np

# Enable OpenCL for GPU acceleration
cv2.ocl.setUseOpenCL(True)
print(f"OpenCL Enabled: {cv2.ocl.useOpenCL()}")
print(f"OpenCL Device: {cv2.ocl.Device.getDefault().name()}")

def process_match(video_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'output_action.mp4')
        
    detector = TennisCourtDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 1. İlk frame'i al ve manuel köşe seçimi yap
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
        
    print("Lutfen acilan pencerede kortun 4 kosesini secin.")
    corners = detector._select_corners_manually(first_frame)
    
    if corners is None:
        print("Köşe seçimi iptal edildi veya başarısız.")
        return
        
    print("Köşeler seçildi, video işleniyor...")
    
    # Referans görüntü oluştur (Aksiyon anı tespiti için)
    _, ref_warped, _, _ = detector.process_video_frame(first_frame, corners=corners)
    
    # Video yazıcı ayarları
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Çıktı olarak orijinal görüntünün işlenmiş halini mi yoksa sadece kuş bakışını mı istediği
    # Kullanıcı isteği: "Ekranı eğip bükmesini istemiyorum orjinal videoda sadece aksiyon harici kısımları kırpsın"
    # Bu yüzden orijinal boyutları kullanacağız ve orijinal frame'i kaydedeceğiz.
    
    out_width = width
    out_height = height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    frame_count = 0
    action_frames = 0
    in_action = False
    action_start_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Sabit köşelerle işle (Sadece benzerlik kontrolü için warped alıyoruz)
        processed, warped, _, _ = detector.process_video_frame(frame, corners=corners)
        
        if warped is not None:
            # Benzerlik kontrolü
            similarity = detector.compare_frames(ref_warped, warped)
            
            # Eşik değer
            is_current_action = similarity > 0.85
            
            # Durum değişikliği kontrolü
            if is_current_action and not in_action:
                print(f"[Frame {frame_count}] ACTION STARTED (Sim: {similarity:.2f})")
                in_action = True
                action_start_frame = frame_count
            elif not is_current_action and in_action:
                duration = frame_count - action_start_frame
                print(f"[Frame {frame_count}] ACTION ENDED (Duration: {duration} frames, Sim: {similarity:.2f})")
                in_action = False
            
            # Periyodik log (Durumu da göster)
            if frame_count % 100 == 0: # Log sıklığını azalttım çünkü transition logları var
                status = "ACTION" if is_current_action else "SKIP"
                print(f"Processing frame {frame_count}... [{status}] (Sim: {similarity:.2f})")

            if is_current_action:
                out.write(frame) # Orijinal frame'i yaz
                action_frames += 1
                
    cap.release()
    out.release()
    print(f"İşlem tamamlandı. Toplam {frame_count} kareden {action_frames} aksiyon karesi kaydedildi.")
    print(f"Çıktı dosyası: {output_path}")

if __name__ == "__main__":
    # Video dosyasının adı
    video_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'bestTennisMatch.mp4')
    process_match(video_file)
