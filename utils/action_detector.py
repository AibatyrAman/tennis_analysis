import cv2
import numpy as np
import matplotlib.pyplot as plt

class TennisCourtDetector:
    def __init__(self):
        # Gerçek kort ölçüleri (metre cinsinden)
        self.court_length = 23.77  # Uzunluk
        self.court_width_doubles = 10.97  # Çiftler için genişlik
        self.court_width_singles = 8.23  # Tekler için genişlik
        self.net_height = 0.914  # File yüksekliği (metre)
        
    def preprocess_image(self, image):
        """Görüntüyü ön işleme tabi tutar"""
        # UMat'a çevir (GPU kullanımı için)
        if not isinstance(image, cv2.UMat):
            image = cv2.UMat(image)
            
        # HLS renk uzayına çevir (Beyaz renk tespiti için L kanalı önemli)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        # Beyaz renk maskesi
        # L kanalı için eşik değeri (0-255). 190-200 iyi bir başlangıç noktasıdır.
        lower_white = np.array([0, 190, 0])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(hls, lower_white, upper_white)
        
        # Maskeyi iyileştir (Morphological operations)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # Blur uygula
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return blurred
    
    def detect_court_lines(self, image):
        """Kort çizgilerini tespit eder"""
        # Ön işleme (UMat döner)
        processed = self.preprocess_image(image)
        
        # Canny kenar tespiti (UMat destekler)
        edges = cv2.Canny(processed, 50, 150, apertureSize=3)
        
        # Hough Line Transform ile çizgileri tespit et
        # HoughLinesP çıktısı numpy array'dir (CPU)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50, # Eşik değerini düşürdük çünkü maske daha temiz
            minLineLength=50,
            maxLineGap=20
        )
        
        # Edges'i CPU'ya geri al (görselleştirme için gerekebilir)
        if isinstance(edges, cv2.UMat):
            edges = edges.get()
            
        # Lines'ı CPU'ya geri al (iterasyon için)
        if isinstance(lines, cv2.UMat):
            lines = lines.get()
            
        return edges, lines
    
    def find_court_corners(self, image, lines):
        """Kort köşelerini bulur"""
        if lines is None:
            return None, None
        
        # Görüntü boyutları
        if isinstance(image, cv2.UMat):
            height, width = image.get().shape[:2]
            line_image = image.get().copy()
        else:
            height, width = image.shape[:2]
            line_image = image.copy()
        
        # Çizgileri çiz (görselleştirme için)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Köşeleri bul
        corners = self.detect_corners_advanced(lines, width, height)
        
        return line_image, corners
    
    def detect_corners_advanced(self, lines, width, height):
        """Gelişmiş köşe tespiti"""
        # Çizgileri yatay ve dikey olarak sınıflandır
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 45 or angle > 135:  # Yatay çizgiler (toleransı artırdık)
                horizontal_lines.append(line[0])
            elif 45 <= angle <= 135:  # Dikey çizgiler
                vertical_lines.append(line[0])
        
        # Kesişim noktalarını bul
        corners = []
        # Daha fazla çizgi dene
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = self.line_intersection(h_line, v_line)
                if intersection is not None:
                    x, y = intersection
                    # Görüntü sınırları içinde mi (biraz toleransla)
                    if -50 <= x < width + 50 and -50 <= y < height + 50:
                        corners.append([x, y])
        
        if len(corners) >= 4:
            corners = np.array(corners)
            
            # Kümeleme (birbirine yakın köşeleri birleştir)
            # Basit bir yöntem: K-Means veya sadece mesafe kontrolü
            # Burada basitçe en dıştaki mantıklı 4 köşeyi seçeceğiz
            
            # Önce sırala
            corners = self.sort_corners(corners)
            
            # Eğer çok fazla köşe varsa, muhtemelen gürültü vardır.
            # En iyi 4'lüyü seçmek için alan kontrolü yapılabilir ama şimdilik
            # en dıştakileri alalım (convex hull mantığına yakın)
            
            # Basitçe ilk 4'ü döndür (sıralanmış olduğu için)
            # Ancak bu her zaman doğru olmayabilir. 
            # İyileştirme: Köşelerin oluşturduğu alanın büyüklüğüne bakılabilir.
            
            return corners[[0, 1, 2, 3]] # En basit hali
        
        return None
    
    def line_intersection(self, line1, line2):
        """İki çizginin kesişim noktasını bulur"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        x = x1 + t * (x2-x1)
        y = y1 + t * (y2-y1)
        
        return (int(x), int(y))
    
    def sort_corners(self, corners):
        """Köşeleri saat yönünde sıralar: sol üst, sağ üst, sağ alt, sol alt"""
        # Tekrarlayan noktaları temizle (basitçe)
        unique_corners = []
        for c in corners:
            is_close = False
            for u in unique_corners:
                if np.linalg.norm(c - u) < 20: # 20 piksel yakınlık
                    is_close = True
                    break
            if not is_close:
                unique_corners.append(c)
        
        corners = np.array(unique_corners)
        if len(corners) < 4:
            return corners # Yeterli köşe yok
            
        # Merkez noktayı hesapla
        center = corners.mean(axis=0)
        
        # Açılara göre sırala
        angles = np.arctan2(corners[:, 1] - center[1], 
                           corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        sorted_corners = corners[sorted_indices]
        
        # Sol-üst'ten başlayacak şekilde kaydır (gerekirse)
        # Genellikle arctan2 -pi ile pi arası verir.
        # Sol üst: -135 (-3pi/4), Sağ üst: -45 (-pi/4), Sağ alt: 45 (pi/4), Sol alt: 135 (3pi/4)
        # Bu sıralama zaten sol-üst, sağ-üst, sağ-alt, sol-alt sırasına yakın olmalı
        
        return sorted_corners

    def apply_perspective_transform(self, image, src_corners, output_size=(1097, 2377)):
        """Perspektif dönüşümü uygular"""
        # Hedef köşeler (yukarıdan bakış)
        dst_corners = np.float32([
            [0, 0],                          # Sol üst
            [output_size[0], 0],             # Sağ üst
            [output_size[0], output_size[1]], # Sağ alt
            [0, output_size[1]]              # Sol alt
        ])
        
        # src_corners float32 olmalı
        src_corners = np.float32(src_corners)
        
        # Perspektif dönüşüm matrisi
        matrix = cv2.getPerspectiveTransform(
            src_corners, 
            dst_corners
        )
        
        # UMat'a çevir
        if not isinstance(image, cv2.UMat):
            image = cv2.UMat(image)
            
        # Dönüşümü uygula
        warped = cv2.warpPerspective(
            image, 
            matrix, 
            output_size
        )
        
        return warped.get(), matrix
    
    def draw_court_measurements(self, image):
        """Kort üzerine ölçüleri çizer"""
        # Eğer UMat ise numpy'a çevir (çizim için garanti olsun)
        if isinstance(image, cv2.UMat):
            image = image.get()
            
        height, width = image.shape[:2]
        
        # Ölçü metni
        cv2.putText(image, f"Uzunluk: {self.court_length}m", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Genislik: {self.court_width_doubles}m", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image
    
    def _select_corners_manually(self, image):
        """Kullanıcının bir görüntüden manuel olarak 4 köşe seçmesini sağlar."""
        # Manuel seçim CPU'da yapılır
        if isinstance(image, cv2.UMat):
            image = image.get()
            
        corners = []
        clone = image.copy()
        window_name = "Manuel Kose Secimi"

        def select_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x, y])
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(clone, str(len(corners)), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(window_name, clone)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_point)

        print("Lutfen kortun 4 kosesini sirasiyla secin (sol-ust, sag-ust, sag-alt, sol-alt) ve ardindan 'c' tusuna basarak onaylayin. 'r' ile sifirlayabilirsiniz.")

        while True:
            cv2.imshow(window_name, clone)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(corners) == 4:
                break
            elif key == ord('r'):
                corners = []
                clone = image.copy()
                print("Secim sifirlandi. Lutfen 4 koseyi tekrar secin.")

        cv2.destroyWindow(window_name)
        cv2.waitKey(1) # Pencerenin kapanmasini bekle

        if len(corners) == 4:
            return np.array(corners, dtype=np.float32)
        return None

    def compare_frames(self, frame1, frame2):
        """İki görüntü arasındaki benzerliği hesaplar (Histogram karşılaştırma)"""
        # UMat'a çevir
        if not isinstance(frame1, cv2.UMat):
            frame1 = cv2.UMat(frame1)
        if not isinstance(frame2, cv2.UMat):
            frame2 = cv2.UMat(frame2)
            
        # Görüntüleri HSV'ye çevir
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        
        # Histogramları hesapla
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        # Normalize et
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX, -1)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX, -1)
        
        # Karşılaştır (Correlation)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return similarity

    def process_video_frame(self, frame, use_manual=False, debug=False, corners=None):
        """Video frame'ini işler. Otomatik, manuel veya sabit köşe tespiti yapabilir."""
        debug_data = {}
        
        # Eğer köşeler dışarıdan verildiyse, tespit adımını atla
        if corners is not None:
            processed_image = frame.copy()
            # Köşeleri çiz
            for i, corner in enumerate(corners):
                cv2.circle(processed_image, tuple(map(int, corner)), 10, (255, 0, 0), -1)
                cv2.putText(processed_image, str(i + 1), tuple(map(int, corner)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # apply_perspective_transform içinde UMat dönüşümü yapılıyor
            warped, matrix = self.apply_perspective_transform(frame, corners)
            warped = self.draw_court_measurements(warped)
            
            return processed_image, warped, corners, debug_data

        # Orijinal boyutları sakla
        orig_h, orig_w = frame.shape[:2]
        
        # İşleme için boyutu küçült (Performans ve gürültü azaltma için)
        target_width = 1000
        scale = 1.0
        working_frame = frame.copy()
        
        if orig_w > target_width:
            scale = orig_w / target_width
            new_h = int(orig_h / scale)
            working_frame = cv2.resize(frame, (target_width, new_h))
            
        # 1. Köşeleri tespit et (manuel veya otomatik)
        # corners zaten None
        processed_image = frame.copy() # Çizimler orijinal üzerine yapılacak

        if use_manual:
            corners = self._select_corners_manually(frame)
        else: # Otomatik
            # Algılamayı küçültülmüş görüntüde yap (detect_court_lines UMat kullanır)
            # working_frame'i UMat'a çevirelim
            working_frame_umat = cv2.UMat(working_frame)
            edges, lines = self.detect_court_lines(working_frame_umat)
            
            if debug:
                debug_data['edges'] = edges
                # Çizgileri görselleştir (küçük resim üzerinde)
                line_img_debug = working_frame.copy()
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(line_img_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                debug_data['lines_debug'] = line_img_debug

            if lines is not None:
                # Köşeleri bul (küçük resim koordinatlarında)
                # find_court_corners CPU'da çalışır (lines numpy array)
                _, corners_small = self.find_court_corners(working_frame, lines)
                
                if corners_small is not None and len(corners_small) == 4:
                    # Koordinatları orijinal boyuta ölçekle
                    corners = corners_small * scale
        
        # 2. Köşeler bulunduysa perspektif dönüşümü yap
        if corners is not None and len(corners) == 4:
            # Köşeleri son görüntü üzerine çiz
            for i, corner in enumerate(corners):
                cv2.circle(processed_image, tuple(map(int, corner)), 10, (255, 0, 0), -1)
                cv2.putText(processed_image, str(i + 1), tuple(map(int, corner)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
            warped, matrix = self.apply_perspective_transform(frame, corners)
            warped = self.draw_court_measurements(warped)
            
            return processed_image, warped, corners, debug_data

        # 3. Başarısızlık durumu
        return frame, None, None, debug_data
