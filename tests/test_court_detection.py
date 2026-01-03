import cv2
import os
import sys
from court_line_detector import CourtLineDetector

def process_input(input_path, model_path, output_path):
    print(f"Model yükleniyor: {model_path}")
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı -> {model_path}")
        return

    # Initialize detector
    detector = CourtLineDetector(model_path)
    
    # Check if image or video
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process Image
        print(f"Resim işleniyor: {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            print("Hata: Resim okunamadı.")
            return

        # --- DEBUG START: Visualize Model Input ---
        # Resize manually to see what the model sees
        debug_img = cv2.resize(image, (224, 224))
        
        # Predict using the detector
        keypoints = detector.predict(image)
        
        # Create a copy of the debug image to draw raw keypoints
        debug_img_draw = debug_img.copy()
        
        # Map keypoints back to 224x224 space for visualization
        # The detector returns keypoints scaled to original image size. 
        # We need to reverse this to see them on the 224x224 image.
        original_h, original_w = image.shape[:2]
        
        raw_keypoints = keypoints.copy()
        raw_keypoints[::2] *= 640.0 / original_w
        raw_keypoints[1::2] *= 640.0 / original_h
        
        for i in range(0, len(raw_keypoints), 2):
            x = int(raw_keypoints[i])
            y = int(raw_keypoints[i+1])
            cv2.putText(debug_img_draw, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            cv2.circle(debug_img_draw, (x, y), 3, (0, 0, 255), -1)
            
        debug_output_path = "debug_model_view_224x224.jpg"
        cv2.imwrite(debug_output_path, debug_img_draw)
        print(f"DEBUG: Modelin gördüğü boyut (224x224) üzerine tahminler kaydedildi: {debug_output_path}")
        # --- DEBUG END ---

        output_image = detector.draw_keypoints(image, keypoints)
        
        cv2.imwrite(output_path, output_image)
        print(f"Çıktı kaydedildi: {output_path}")
        
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process Video
        print(f"Video işleniyor: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Hata: Video açılamadı.")
            return

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        print(f"Toplam kare sayısı: {len(frames)}")
        
        # Process frames
        output_frames = detector.draw_keypoints_on_video(frames)
        
        # Save video
        if output_frames:
            height, width, _ = output_frames[0].shape
            # codec = cv2.VideoWriter_fourcc(*'mp4v')
            codec = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, codec, fps, (width, height))
            
            for frame in output_frames:
                out.write(frame)
            out.release()
            print(f"Video kaydedildi: {output_path}")
    else:
        print("Desteklenmeyen dosya formatı.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python test_court_detection.py <input_path> [output_path] [model_path]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"Hata: Girdi dosyası bulunamadı -> {input_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_detected{ext}"

    if len(sys.argv) >= 4:
        model_path = sys.argv[3]
    else:
        model_path = "models/keypoints_model_50.pt"

    process_input(input_path, model_path, output_path)
