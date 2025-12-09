# Tenis Maçı Analiz Projesi

Bu proje, tenis maçlarındaki oyuncuları ve topu tespit edip takip etmek, ayrıca tenis kortu çizgilerini belirlemek için geliştirilmiş bir yapay zeka uygulamasıdır. YOLO (You Only Look Once) ve CNN (Convolutional Neural Networks) modellerini kullanır.

## Özellikler

- **Oyuncu Takibi:** YOLOv8 kullanarak sahadaki oyuncuları tespit eder ve takip eder.
- **Top Takibi:** Tenis topunu tespit eder ve hareketini izler.
- **Kort Çizgisi Tespiti:** Önceden eğitilmiş bir ResNet50 modeli kullanarak kortun önemli noktalarını ve çizgilerini belirler.
- **Hız ve İstatistikler:** (Gelecekte eklenebilir) Oyuncu hızı ve istatistikleri çıkarılabilir.

## Gereksinimler

Projeyi çalıştırmak için Python 3.8 veya daha yeni bir sürüme ihtiyacınız vardır. Ayrıca aşağıdaki kütüphanelerin yüklü olması gerekir:

- `ultralytics` (YOLO modelleri için)
- `opencv-python` (Görüntü işleme için)
- `torch` ve `torchvision` (Derin öğrenme modelleri için)
- `numpy` (Matematiksel işlemler için)

## Git LFS Kurulumu

Bu proje, büyük model dosyalarını (.pt) yönetmek için **Git LFS (Large File Storage)** kullanmaktadır. `.gitattributes` dosyasında `.pt` uzantılı dosyalar LFS ile yönetilmek üzere yapılandırılmıştır.

### Git LFS Kurulumu

Projeyi klonlamadan veya çalıştırmadan önce Git LFS'in kurulu olması gerekmektedir:

#### macOS

```bash
# Homebrew ile kurulum
brew install git-lfs

# Git LFS'i başlatma
git lfs install
```

#### Linux (Ubuntu/Debian)

```bash
# APT ile kurulum
sudo apt-get update
sudo apt-get install git-lfs

# Git LFS'i başlatma
git lfs install
```

#### Windows

1. [Git LFS resmi sitesinden](https://git-lfs.github.com/) installer'ı indirin
2. İndirilen `.exe` dosyasını çalıştırarak kurulumu tamamlayın
3. Terminalde aşağıdaki komutu çalıştırın:

```bash
git lfs install
```

### Projeyi Klonlama

Git LFS kurulumundan sonra projeyi klonladığınızda, LFS dosyaları otomatik olarak indirilecektir:

```bash
git clone <repository-url>
cd tennis_analysis
```

Eğer projeyi daha önce klonladıysanız ve LFS dosyaları indirilmediyse, aşağıdaki komutu çalıştırın:

```bash
git lfs pull
```

## Kurulum

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu terminalde çalıştırabilirsiniz:

```bash
pip install ultralytics opencv-python torch torchvision numpy
```

## Dosya Yapısı ve Açıklamalar

Projedeki ana dosya ve klasörlerin görevleri şunlardır:

- **`main.py`**: Projenin giriş noktasıdır. Videoyu okur, takip algoritmalarını (trackers) ve çizgi tespitini çalıştırır, sonuçları işleyerek çıktı videosunu oluşturur.
- **`trackers/`**:
  - `player_tracker.py`: Oyuncuları tespit etmek ve takip etmek için gerekli sınıfları içerir.
  - `ball_tracker.py`: Topu tespit etmek için özelleştirilmiş mantığı içerir.
- **`court_line_detector/`**:
  - `court_line_detector.py`: Eğitilmiş bir CNN modeli (ResNet50) kullanarak kortun köşe noktalarını tespit eder.
- **`utils/`**:
  - `video_utils.py`: Video okuma ve kaydetme gibi yardımcı fonksiyonları barındırır.
- **`models/`**: Projenin kullandığı eğitilmiş model dosyalarını (.pt veya .pth) içerir.
- **`input_videos/`**: İşlenecek ham videoların konulacağı klasör.
- **`output_videos/`**: İşlenmiş ve üzerine çizim yapılmış videoların kaydedildiği klasör.
- **`tracker_stubs/`**: Tespit işlemlerini her seferinde tekrar yapmamak için sonuçların pkl formatında kaydedildiği önbellek klasörü.

## Nasıl Çalıştırılır?

1. Analiz etmek istediğiniz videoyu `input_videos` klasörüne kopyalayın (örneğin `input_video.mp4`).
2. `main.py` dosyasını açarak `input_video_path` değişkeninin dosya adınızla eşleştiğinden emin olun.
3. Terminalde aşağıdaki komutu çalıştırın:

```bash
python main.py
```

4. İşlem tamamlandığında, sonuç videosunu `output_videos` klasöründe bulabilirsiniz.

## Notlar

- İlk çalıştırmada YOLO modellerinin indirilmesi biraz zaman alabilir.
- `tracker_stubs` parametresi `True` ise, kod daha önceki çalıştırmalardaki tespitleri kullanır. Yeni bir video için bu özelliği kapatmanız veya stubs klasörünü temizlemeniz gerekebilir.
