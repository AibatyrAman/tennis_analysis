from ultralytics import YOLO
import cv2
import pickle
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trackers import BallTracker
from court_line_detector import CourtLineDetector
from utils.device_utils import get_best_device

class MultiBallTracker(BallTracker):
    """
    Modified BallTracker for testing:
    1. Multiple Object Detection: Stores ALL detected balls in a frame.
    2. No Interpolation: Skips interpolation logic.
    3. Court Filtering: Ignored balls outside the court polygon.
    """
    def __init__(self, model_path, court_polygon=None):
        super().__init__(model_path)
        self.court_polygon = court_polygon

    def detect_frame(self, frame):
        device = get_best_device()
        # Ensure we detect all balls with reasonable confidence
        results = self.model.predict(frame, conf=0.15, device=device)[0]

        ball_dict = {}
        # 1. Multiple Ball Detection Logic
        for i, box in enumerate(results.boxes):
            result = box.xyxy.tolist()[0]
            x1, y1, x2, y2 = result
            center = (int((x1+x2)/2), int((y1+y2)/2))
            
            # Court Filtering
            if self.court_polygon is not None:
                # Check if center is inside (or on edge) of the court polygon
                # measureDist=False returns +1 (inside), -1 (outside), 0 (on edge)
                is_inside = cv2.pointPolygonTest(self.court_polygon, center, False)
                if is_inside < 0:
                    continue # Skip this ball

            ball_dict[i+1] = result 
        
        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes for ALL detected balls
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                # Visual distinction: Draw Yellow Box
                cv2.putText(frame, f"Ball {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            
            # Draw Court Polygon for verification if exists
            if hasattr(self, 'court_polygon') and self.court_polygon is not None:
                cv2.polylines(frame, [self.court_polygon], True, (0, 0, 255), 2)
                
            output_video_frames.append(frame)
        
        return output_video_frames

def run_test(input_path, model_path="models/yeni_model.pt", court_model_path="models/keypoints_model_50.pt"):
    print(f"--- ÇOKLU TOP TAKİBİ TESTİ (KORT FİLTRELİ) ---")
    print(f"Girdi: {input_path}")
    print(f"Model: {model_path}")
    print(f"Kort Model: {court_model_path}")

    # 1. Initialize Court Detector
    court_detector = CourtLineDetector(court_model_path)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Video açılamadı!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Read first frame for court detection
    ret, first_frame = cap.read()
    if not ret:
        print("İlk kare okunamadı.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset
    
    # Detect Court Keypoints
    print("Kort tespit ediliyor (İlk Kare)...")
    court_keypoints = court_detector.predict(first_frame)
    
    # Create Court Polygon (Convex Hull)
    xs = court_keypoints[::2]
    ys = court_keypoints[1::2]
    points = np.column_stack((xs, ys)).astype(np.int32)
    hull = cv2.convexHull(points)
    
    def scale_polygon(polygon, scale_x=1.2, scale_y=1.2):
        # Calculate centroid
        moments = cv2.moments(polygon)
        if moments['m00'] == 0: return polygon
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Scale points
        cnt_norm = polygon - [cx, cy]
        # Multiply x by scale_x, y by scale_y
        cnt_scaled = cnt_norm * [scale_x, scale_y]
        cnt_final = cnt_scaled + [cx, cy]
        return cnt_final.astype(np.int32)

    # Add Buffer: 
    # X (Width): 1.1 (User manually set this)
    # Y (Length): 1.1 * 1.05 = 1.155 (User requested +5% vertically)
    court_polygon = scale_polygon(hull, scale_x=1.1, scale_y=1.3)
    
    print("Kort Poligonu oluşturuldu (Buffer eklendi).")

    # 2. Initialize Ball Tracker with Polygon
    ball_tracker = MultiBallTracker(model_path=model_path, court_polygon=court_polygon)

    output_path = f"tests/output_multiball_filtered_{os.path.basename(input_path)}"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames = []
    MAX_FRAMES = 500
    print(f"İlk {MAX_FRAMES} kare işlenecek...")

    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= MAX_FRAMES:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("Hiç kare okunamadı.")
        return

    # 3. Detect Balls
    print("Toplar tespit ediliyor (Kort Filtreli)...")
    ball_detections = ball_tracker.detect_frames(frames, read_from_stub=False)

    # 4. Draw Results
    print("Sonuçlar çiziliyor...")
    output_frames = ball_tracker.draw_bboxes(frames, ball_detections)

    # 5. Write Video
    for f in output_frames:
        out.write(f)
    
    out.release()
    print(f"Test tamamlandı: {output_path}")

if __name__ == "__main__":
    input_file = "input_videos/input_video.mp4"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if os.path.exists(input_file):
        run_test(input_file)
    elif os.path.exists("input_videos/input_video.mp4"):
        print(f"Input {input_file} bulunamadı, default video kullanılıyor.")
        run_test("input_videos/input_video.mp4")
    else:
        print(f"Video bulunamadı: {input_file}")
