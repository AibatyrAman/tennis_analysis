import cv2
import os
import sys
import numpy as np

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mini_court import MiniCourt
import constants

def test_mini_court_logic():
    print("--- MİNİ KORT MATEMATİK TESTİ ---")
    
    # 1. Dummy Frame oluştur
    # (MiniCourt sınıfı init için bir frame istiyor, boyutları almak için)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mini_court = MiniCourt(dummy_frame)
    
    print("Mini Court padding değerleri:")
    print(f"X Padding: {mini_court.padding_court}")
    print(f"Y Padding: {mini_court.buffer}")
    print(f"Drawing Width: {mini_court.drawing_rectangle_width}")
    print(f"Drawing Height: {mini_court.drawing_rectangle_height}")

    # 2. Test Noktaları (Pixel -> Metre Dönüşümü)
    # Örnek: Kortun tam ortası (piksel olarak değil, mantıksal olarak test edelim)
    # Bu test matematiksel dönüşüm fonksiyonlarını direkt çağırmadığı için
    # (çünkü o fonksiyonlar main.py veya class içinde gömülü)
    # Burada çizim yeteneğini test edeceğiz.
    
    print("\n--- Çizim Testi ---")
    # Boş bir tuval üzerine mini kort çizdir
    # Normalde main.py'de 350x600 boyutlarında bir tuval kullanmıştık.
    test_width = 350
    test_height = 600
    canvas = np.zeros((test_height, test_width, 3), dtype=np.uint8)
    
    # Bu tuvale uygun yeni bir MiniCourt nesnesi (Main.py'deki perfect_mini_court mantığı)
    perfect_mini_court = MiniCourt(canvas)
    
    # Çiz
    frames = [canvas.copy()] # List olarak bekliyor
    output_frames = perfect_mini_court.draw_mini_court(frames)
    
    # Rastgele bir oyuncu ve top ekleyelim
    # Frame 0 için
    # Oyuncu 1: x=50, y=50 (Mini kort koordinat sisteminde değil, pixel olarak çizdiriyoruz)
    # Not: draw_points_on_mini_court fonksiyonu, convert edilmiş koordinatları bekler.
    
    # Simüle edilmiş "Dönüştürülmüş" koordinatlar:
    # Mini kortun (350x600) içinde mantıklı yerler:
    # Merkez yaklaşık: (175, 300)
    # Sol üst köşe padding dahilinde
    
    mock_player_positions = [
        {1: (175, 550), 2: (175, 50)} # Oyuncular karşılıklı
    ]
    mock_ball_positions = [
        {1: (175, 300)} # Top tam file üstünde
    ]
    
    print("Oyuncu ve Top çiziliyor...")
    output_frames = perfect_mini_court.draw_points_on_mini_court(output_frames, mock_player_positions, color=(0,0,255)) # Kırmızı Oyuncu
    output_frames = perfect_mini_court.draw_points_on_mini_court(output_frames, mock_ball_positions, color=(0,255,255)) # Sarı Top
    
    output_path = "tests/output_mini_court_test.jpg"
    cv2.imwrite(output_path, output_frames[0])
    print(f"Mini kort test çıktısı kaydedildi: {output_path}")
    print("Bu resmi açıp kortun, çizgilerin ve oyuncu noktalarının düzgün görünüp görünmediğini kontrol edin.")

if __name__ == "__main__":
    test_mini_court_logic()
