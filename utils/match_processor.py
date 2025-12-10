import cv2
import os
import sys
from .action_detector import TennisCourtDetector

def process_match(video_path, output_path):
    """
    Video içerisindeki aksiyon anlarını tespit eder ve sadece bu anları içeren yeni bir video oluşturur.
    
    Args:
        video_path (str): Girdi video dosyasının yolu.
        output_path (str): Çıktı video dosyasının kaydedileceği yol.
        
    Returns:
        str: Oluşturulan videonun yolu.
    """
    
    # GPU hızlandırmasını etkinleştir
    cv2.ocl.setUseOpenCL(True)
    print(f"OpenCL Enabled: {cv2.ocl.useOpenCL()}")

    detector = TennisCourtDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # 1. İlk frame'i al ve manuel köşe seçimi yap
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return None
        
    print("Lutfen acilan pencerede kortun 4 kosesini secin.")
    corners = detector._select_corners_manually(first_frame)
    
    if corners is None:
        print("Köşe seçimi iptal edildi veya başarısız.")
        return None
        
    print("Köşeler seçildi, video işleniyor... (Bu işlem biraz zaman alabilir)")
    
    # Referans görüntü oluştur (Aksiyon anı tespiti için)
    # corners parametresi ile sabit köşe tespiti yapıyoruz
    _, ref_warped, _, _ = detector.process_video_frame(first_frame, corners=corners)
    
    # Video yazıcı ayarları
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
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
        # corners parametresi burada da verilmeli
        processed, warped, _, _ = detector.process_video_frame(frame, corners=corners)
        
        if warped is not None:
            # Benzerlik kontrolü
            similarity = detector.compare_frames(ref_warped, warped)
            
            # Eşik değer - Aksiyon mu?
            is_current_action = similarity > 0.85
            
            # Durum değişikliği kontrolü ve loglama
            if is_current_action and not in_action:
                print(f"[Frame {frame_count}] ACTION STARTED (Sim: {similarity:.2f})")
                in_action = True
                action_start_frame = frame_count
            elif not is_current_action and in_action:
                duration = frame_count - action_start_frame
                # Çok kısa aksiyonları loglamayabiliriz ama şimdilik kalsın
                if duration > 10: 
                     print(f"[Frame {frame_count}] ACTION ENDED (Duration: {duration} frames)")
                in_action = False
            
            # Periyodik ilerleme göstergesi
            if frame_count % 500 == 0:
                print(f"Processed {frame_count} frames...")

            if is_current_action:
                out.write(frame) # Orijinal frame'i yaz
                action_frames += 1
                
    cap.release()
    out.release()
    print(f"Aksiyon filtreleme tamamlandı. Toplam {frame_count} kareden {action_frames} aksiyon karesi kaydedildi.")
    print(f"Filtrelenmiş video: {output_path}")
    
    return output_path, corners
