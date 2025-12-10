# Tenis Maçı Analiz Projesi

Bu proje, tenis maçlarındaki oyuncuları ve topu tespit edip takip etmek, ayrıca tenis kortu çizgilerini belirlemek için geliştirilmiş bir yapay zeka uygulamasıdır. YOLO (You Only Look Once), CNN (Convolutional Neural Networks) ve Görüntü İşleme tekniklerini kullanır.

## Özellikler

- **Aksiyon Filtreleme:** Uzun maç videolarındaki sadece oyunun olduğu aksiyon anlarını tespit eder ve ayıklar.
- **Oyuncu Takibi:** YOLOv8 kullanarak sahadaki oyuncuları tespit eder ve takip eder.
- **Top Takibi:** Tenis topunu tespit eder, yörüngesini tamamlar (interpolation) ve hareketini izler.
- **Kort Çizgisi Tespiti:** Önceden eğitilmiş bir ResNet50 modeli kullanarak kortun önemli noktalarını ve çizgilerini belirler.
- **Sekme Tespiti (Bounce Detection):** Topun yere değdiği anları analiz eder.
- **Isı Haritası (Heatmap):** Topun sektigi noktalari 2D mini kort uzerinde isi haritasi olarak gorsellestirir.

## Gereksinimler

Projeyi çalıştırmak için Python 3.8 veya daha yeni bir sürüme ihtiyacınız vardır. Ayrıca aşağıdaki kütüphanelerin yüklü olması gerekir:

- `ultralytics` (YOLO modelleri için)
- `opencv-python` (Görüntü işleme için)
- `torch` ve `torchvision` (Derin öğrenme modelleri için)
- `numpy` (Matematiksel işlemler için)
- `pandas` (Veri interpolasyonu ve analizi için)
- `scipy` (Sinyal işleme ve sekme tespiti için)

## Kurulum

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu terminalde çalıştırabilirsiniz:

```bash
pip install ultralytics opencv-python torch torchvision numpy pandas scipy
```

## Dosya Yapısı ve Açıklamalar

Projedeki ana dosya ve klasörlerin görevleri şunlardır:

- **`main.py`**: Projenin giriş noktasıdır. Aksiyon filtresini, takip algoritmalarını ve görselleştirmeyi yönetir.
- **`trackers/`**:
  - `player_tracker.py`: Oyuncuları tespit etmek ve takip etmek için gerekli sınıfları içerir.
  - `ball_tracker.py`: Topu tespit etmek, interpolasyon yapmak ve sekme anlarını bulmak için özelleştirilmiş mantığı içerir.
- **`court_line_detector/`**:
  - `court_line_detector.py`: Eğitilmiş bir CNN modeli (ResNet50) kullanarak kortun köşe noktalarını tespit eder.
- **`utils/`**:
  - `video_utils.py`: Video okuma ve kaydetme gibi yardımcı fonksiyonları barındırır.
  - `match_processor.py`: Video içerisindeki aksiyon anlarını filtreler.
  - `action_detector.py`: Kort tespiti ve perspektif dönüşümü için gerekli temel sınıfları içerir.
  - `mini_court.py`: 2D mini kort çizimi, koordinat dönüşümü ve ısı haritası oluşturma işlemlerini yapar.
- **`models/`**: Projenin kullandığı eğitilmiş model dosyalarını (.pt veya .pth) içerir.
- **`input_videos/`**: İşlenecek ham videoların konulacağı klasör.
- **`output_videos/`**: İşlenmiş (filtrelenmiş ve analiz edilmiş) videoların kaydedildiği klasör.
- **`tracker_stubs/`**: Tespit işlemlerini her seferinde tekrar yapmamak için sonuçların pkl formatında kaydedildiği önbellek klasörü.

## Nasıl Çalıştırılır?

1. Analiz etmek istediğiniz videoyu `input_videos` klasörüne kopyalayın (örneğin `input_video.mp4`).
2. `main.py` dosyasını açarak `input_video_path` değişkeninin dosya adınızla eşleştiğinden emin olun.
3. Terminalde aşağıdaki komutu çalıştırın:

```bash
python main.py
```

4. **Manuel Köşe Seçimi:** Kod çalıştırıldığında bir pencere açılacak ve sizden kortun 4 köşesini seçmeniz istenecektir. Sırasıyla **Sol-Üst, Sağ-Üst, Sağ-Alt, Sol-Alt** köşelerini seçin ve 'c' tuşuna basarak onaylayın.
5. İşlem tamamlandığında:
   - Aksiyon sahneleri ayrıştırılmış video `output_videos/filtered_action.mp4` olarak kaydedilir.
   - Analiz edilmiş ve görselleştirilmiş son video `output_videos/output_video.mp4` olarak kaydedilir.

## Notlar

- İlk çalıştırmada YOLO modellerinin indirilmesi biraz zaman alabilir.
- `tracker_stubs` parametresi `True` ise, kod daha önceki çalıştırmalardaki tespitleri kullanır. Yeni bir video için bu özelliği kapatmanız veya stubs klasörünü temizlemeniz gerekebilir.
- Aksiyon filtresi ve kort dönüşümü için manuel köşe seçimi kritiktir, lütfen köşeleri dikkatli seçin.
