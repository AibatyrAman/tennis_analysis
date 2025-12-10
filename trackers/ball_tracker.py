from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            players_dict = self.detect_frame(frame)
            ball_detections.append(players_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections



    def detect_frame(self, frame):
        results = self.model.predict(frame,conf=0.15)[0] # sadece tespit değil takip de yapıyoruz ve id lerin özel olmasını sağlıyoruz
        
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict
    
    
    def interpolate_ball_positions(self, ball_positions):
        import pandas as pd
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() # Fill first frames if missing

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff() # Change in Y
        
        # Bounce logic: The ball hits the ground when Y is maximized (locally)
        # However, physics: Y increases (goes down), hits ground, Y decreases (goes up).
        # So we look for local MAXIMA in Y coordinate.
        # Simple loop to find local max
        
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            current_y = df_ball_positions.iloc[i]['mid_y']
            prev_y = df_ball_positions.iloc[i-1]['mid_y']
            next_y = df_ball_positions.iloc[i+1]['mid_y']

            # Check if it's a peak (local max) -> This means lowest point on screen -> Bounce
            if current_y > prev_y and current_y > next_y:
                # Filter noise: verify it keeps going up (decreasing Y) after and came down (increasing Y) before
                # Looking at rolling mean might be better?
                pass
                
        # Let's try a simpler approach: finding peaks in the signal
        # Since we want "bounces", let's assume they are the peaks of Y value.
        
        # Scipy signal finding
        from scipy.signal import find_peaks
        # mid_y is distance from top. So bigger y = lower on screen. Bounce is a PEAK in Y.
        
        # We need to handle potential NaNs if interpolation failed (shouldn't happen with bfill)
        y_values = df_ball_positions['mid_y_rolling_mean'].fillna(0).to_numpy()
        
        # Find peaks (bounces)
        # distance: minimum frames between bounces
        peaks, _ = find_peaks(y_values, distance=30) 
        
        # Verify peaks: check if Y changes significantly around the peak
        # Or just trust find_peaks for now.
        
        return peaks.tolist()

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame,ball_dict in zip(video_frames, player_detections):
            # frame üzerine kutuları çiz
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 =  bbox
                cv2.putText(frame, f"Top ID: {track_id}", (int(bbox[0]) , int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
                