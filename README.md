# Tenis Maçı Analiz Sistemi

Bu proje, tenis maç videolarından kort çizgilerini, oyuncuları, topu ve hareket akışını otomatik olarak tespit ederek maç analizi yapmaya odaklanan bir bilgisayar görüşü uygulamasıdır. Proje, videoyu yükledikten sonra anotasyonlu bir ana video ve mini kort tekrarını üretecek şekilde tasarlanmıştır.

Amaç, tenis videolarından yalnızca görsel verileri değil; aynı zamanda maçın akışını, topun hareketini, oyuncu konumlarını ve temel istatistikleri de anlamlı bir biçimde çıkarmaktır.

---

## Proje Hakkında

Bu sistem aşağıdaki adımları otomatik olarak yürütür:

1. Videoyu yükler.
2. Kort çizgilerini otomatik olarak tespit eder.
3. Oyuncu ve top takibi yapar.
4. Aksiyonlu kareleri filtreleyerek analiz sürecini daha verimli hale getirir.
5. Mini kort üzerinde top ve oyuncu hareketlerini görselleştirir.
6. Sonuçları kullanıcıya bir arayüz üzerinden sunar.

Bu yapı, özellikle tenis videoları üzerinde maç analizi, tekrar görüntüleme ve istatistik çıkarımı yapmak isteyen kullanıcılar için uygundur.

---

## Ana Özellikler

- Otomatik kort tespiti
- Oyuncu takibi
- Top takibi
- Poz takibi
- Aksiyon filtresi ile gereksiz karelerin azaltılması
- Mini kort tabanlı görselleştirme
- İstatistiksel özet çıkarımı
- Flet tabanlı kullanıcı arayüzü
- Opsiyonel skor tablosu seçimi
- Üretilen sonuç videolarının kaydedilmesi

---

## Sistem Akışı

Proje genel olarak şu akışla çalışır:

1. Video girişi alınır.
2. Referans kare üzerinden kort geometrisi çıkarılır.
3. Kort alanı ve hareketli bölgeler belirlenir.
4. Oyuncu, top ve poz tespiti yapılır.
5. Topun yere çarpma noktaları, şutlar ve maç akışı bilgileri çıkarılır.
6. Sonuçlar hem arayüzde hem de çıktı videolarında görüntülenir.

Bu yaklaşım, maçın görsel verisini analiz ederek hem görsel bir tekrar hem de teknik bir değerlendirme sunmayı hedefler.

---

## Kurulum

### Gereksinimler

- Python 3.10 veya üzeri
- OpenCV
- PyTorch
- TorchVision
- Ultralytics
- Pandas
- NumPy
- EasyOCR
- Flet

### Paketlerin Kurulumu

Aşağıdaki komut ile temel bağımlılıkları kurabilirsiniz:

```bash
pip install ultralytics opencv-python pandas numpy torch torchvision easyocr flet
```

> Not: PyTorch sürümü sisteminize göre (CPU/GPU) farklı olabilir. Özellikle NVIDIA GPU kullanan sistemlerde CUDA uyumlu bir PyTorch kurulumu tercih edilmelidir.

---

## Çalıştırma

### 1) Arayüzü Başlatma

Projenin grafik arayüzü şu komutla çalıştırılır:

```bash
python app.py
```

Bu komut çalıştıktan sonra:

- Bir tenis maçı videosu yükleyebilirsiniz.
- İsterseniz skor tablosunu seçebilirsiniz.
- Analizi başlatabilirsiniz.

### 2) Analiz Süreci

Analiz başladıktan sonra proje şu adımları otomatik olarak uygular:

- Video bilgileri okunur.
- Kort tespiti yapılır.
- Aksiyon filtresi uygulanır.
- Takip algoritmaları çalıştırılır.
- İstatistikler hesaplanır.
- Sonuç videoları oluşturulur.

---

## Çıktılar

Analiz tamamlandığında proje aşağıdaki türde çıktı dosyaları üretir:

- Ana analiz videosu
- Mini kort tekrar videosu
- Analiz akışı bilgileri
- Görsel olarak zenginleştirilmiş maç tekrarları

Çıktılar varsayılan olarak proje kök dizinindeki şu klasöre kaydedilir:

```bash
output_videos/
```

---

## Proje Klasör Yapısı

```text
.
├── app.py                  # Flet tabanlı kullanıcı arayüzü
├── main.py                 # Ana analiz pipeline'ı
├── inspect_model.py        # Model yapısını kontrol etmek için yardımcı script
├── constants/              # Sabit değerler ve yapılandırma
├── court_line_detector/    # Kort çizgisi tespiti ve doğrulama
├── mini_court/             # Mini kort görselleştirme mantığı
├── models/                 # Önceden eğitilmiş model ağırlıkları
├── trackers/               # Oyuncu/top/poz/skor takipleyicileri
├── utils/                  # Yardımcı fonksiyonlar
├── tests/                  # Test senaryoları
├── input_videos/           # Girdi videoları için klasör
├── output_videos/          # Üretilen çıktı videoları
└── readme_videos/          # README içinde demo amaçlı kullanılan videolar
```

---

## Demo Videolar

README içinde demo veya tanıtım amaçlı kullanılabilecek videolar şunlardır:

- [Topun Sektiği Noktalar](readme_videos/video_tekrari.mov)
- [İnteraktif Bir Şekilde Topun Sektiği Noktalar Ve Sayı Alınan Noktalar](readme_videos/interaktif.mov)

Bu videolar, sistemin kullanıcı arayüzünden nasıl çalıştığını ve analiz çıktılarının nasıl üretildiğini göstermek için kullanılabilir.

---

## Kullanım Notları

- Daha net ve yüksek çözünürlüklü videolar, tespit başarısını artırır.
- Kötü aydınlatma veya dar açıdan çekilmiş videolarda doğruluk düşebilir.
- Skor tablosu seçimi isteğe bağlıdır; sistem varsayılan olarak otomatik akışa devam edebilir.
- GPU kullanımı, analiz süresini önemli ölçüde azaltabilir.

---

## Testler

Projede yer alan testler şu klasörde bulunur:

```bash
tests/
```

Örnek testler arasında kort tespiti, top takibi, oyuncu takibi, mini kort görselleştirme ve skor tablosu analizi yer alır.

---

## Geliştirme İpuçları

Bu proje daha da geliştirilebilir:

- Daha güçlü skor tablosu OCR desteği eklenebilir.
- Topun atış türü sınıflandırması iyileştirilebilir.
- Daha detaylı oyuncu istatistikleri çıkarılabilir.
- JSON/CSV tabanlı rapor çıktıları eklenebilir.
- Daha iyi kort doğrulama ve hata yönetimi eklenebilir.

---

## Kısa Özet

Bu proje, tenis videolarından maç analizi yapmak için bilgisayar görüşü ve yapay zekâ tabanlı bir akış sunar. Kullanıcı dostu bir arayüz ile videoyu yükleyip analiz ettikten sonra, maçın görsel tekrarını ve temel istatistiksel özetini elde edebilirsiniz.


---
## Vizyonumuz

Bu projede ilerleryen zamanlarda bu bilgileri alıp oyuncular hakkında veri seti oluşturup bu oluşturulan veri setleri üzerinde maç tahmini, karşı takım analizi, antrenör yardımcısı gibi alanlarda ilerletmek istiyoruz.
