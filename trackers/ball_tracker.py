from ultralytics import YOLO 
import cv2
import pickle
import numpy as np
import pandas as pd
from utils.device_utils import get_best_device

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.court_polygon = None

    def set_court_polygon(self, polygon):
        self.court_polygon = polygon

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, [np.nan]*4) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def get_ball_bounce_frames(self, ball_positions):
        """
        Detects frames where the ball bounces on the court (Local Maxima in Y).
        """
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        df['frame'] = df.index
        
        # Calculate centroids
        df['center_y'] = (df['y1'] + df['y2'])/2
        df['center_x'] = (df['x1'] + df['x2'])/2
        
        # We look for local maxima in Y (screen coordinates: Y increases downwards)
        # A bounce is when the ball reaches the lowest point on screen (highest Y) and goes up.
        
        # Smooth the signal to remove jitter
        df['smooth_y'] = df['center_y'].rolling(window=5, center=True).mean()
        
        # Find local maxima (points greater than neighbors)
        # Use a window of 5 frames
        df['is_bounce'] = (df['smooth_y'] > df['smooth_y'].shift(1)) & \
                          (df['smooth_y'] > df['smooth_y'].shift(-1)) & \
                          (df['smooth_y'] > df['smooth_y'].shift(2)) & \
                          (df['smooth_y'] > df['smooth_y'].shift(-2))
        
        # Filter for valid bounces (must have data)
        bounce_frames = df[df['is_bounce'] & df['center_y'].notna()]
        
        return bounce_frames[['frame', 'center_x', 'center_y']].values.tolist()

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        device = get_best_device()
        results = self.model.predict(frame,conf=0.15, device=device)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            
            # Court Filtering
            if self.court_polygon is not None:
                x1, y1, x2, y2 = result
                center = (int((x1+x2)/2), int((y1+y2)/2))
                is_inside = cv2.pointPolygonTest(self.court_polygon, center, False)
                if is_inside < 0:
                    continue 

            # Single Ball Logic: just take the last valid one found (or we could pick max conf)
            # Original code behaviour was overwriting ball_dict[1]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    