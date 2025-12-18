import constants
from copy import deepcopy
import pandas as pd
import numpy as np
import cv2
import os
from utils import (read_video, save_video, measure_distance ,convert_pixel_distance_to_meters)
from utils.player_stats_drawer_utils import draw_player_stats
from utils.manual_selector import select_corners_manually
from trackers import PlayerTracker, BallTracker, PoseTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from utils.action_filter import ActionFilter

class TennisMatchProcessor:
    def __init__(self, model_path_yolo='models/yolov8x.pt', model_path_ball='models/yeni_model.pt', model_path_court='models/keypoints_model_50.pth', model_path_pose='models/yolo11x-pose.pt'):
        self.player_tracker = PlayerTracker(model_path=model_path_yolo)
        self.ball_tracker = BallTracker(model_path=model_path_ball)
        self.pose_tracker = PoseTracker(model_path=model_path_pose)
        self.court_line_detector = CourtLineDetector(model_path_court)

    def get_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        return None

    def process_match(self, input_video_path, corners=None, progress_callback=None):
        if not os.path.exists("output_videos"):
            os.makedirs("output_videos")
            
        base_name = os.path.basename(input_video_path)
        output_video_path = f"output_videos/output_{base_name}"
        output_mini_court_video_path = f"output_videos/mini_court_{base_name}"

        # 1. Read Video
        if progress_callback: progress_callback("Video okunuyor...")
        video_frames, fps = read_video(input_video_path)

        # 2. Action Filtering
        if len(video_frames) > 0:
            if progress_callback: progress_callback("Aksiyon harici kareler filtreleniyor...")
            action_filter = ActionFilter(video_frames[0], roi_corners=corners, similarity_threshold=0.6)
            video_frames = action_filter.filter_frames(video_frames)

        if not video_frames:
            print("Filtreleme sonrası kare kalmadı!")
            return None, None, None

        # 3. Tracking
        if progress_callback: progress_callback("Oyuncular takip ediliyor...")
        player_detections = self.player_tracker.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/player_detections.pkl")
        
        if progress_callback: progress_callback("Top takip ediliyor...")
        ball_detections = self.ball_tracker.detect_frames(video_frames)
        ball_detections = self.ball_tracker.interpolate_ball_positions(ball_detections)

        if progress_callback: progress_callback("İskelet takibi yapılıyor...")
        pose_detections = self.pose_tracker.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/pose_detections.pkl")

        # 4. Court Detection
        if progress_callback: progress_callback("Kort çizgileri tespit ediliyor...")
        court_keypoints = self.court_line_detector.predict(video_frames[0])

        # 5. Filter Players
        player_detections = self.player_tracker.choose_and_filter_players(court_keypoints, player_detections)

        # 6. Mini Court & Stats
        if progress_callback: progress_callback("İstatistikler hesaplanıyor...")
        mini_court = MiniCourt(video_frames[0])
        ball_shot_frames = self.ball_tracker.get_ball_shot_frames(ball_detections)
        
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )

        player_stats_data = self._calculate_stats(ball_shot_frames, ball_mini_court_detections, player_mini_court_detections, fps, mini_court)
        
        # DataFrame Processing
        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
        player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
        player_stats_data_df = player_stats_data_df.ffill()

        # Calculate Averages
        for p in [1, 2]:
            shots = player_stats_data_df[f'player_{p}_number_of_shots']
            player_stats_data_df[f'player_{p}_average_shot_speed'] = player_stats_data_df[f'player_{p}_total_shot_speed'] / shots.replace(0, 1)
            
            opponent = 2 if p == 1 else 1
            opp_shots = player_stats_data_df[f'player_{opponent}_number_of_shots']
            player_stats_data_df[f'player_{p}_average_player_speed'] = player_stats_data_df[f'player_{p}_total_player_speed'] / opp_shots.replace(0, 1)

        # 7. Drawing
        if progress_callback: progress_callback("Video oluşturuluyor...")
        output_video_frames = video_frames.copy()
        output_video_frames = self.player_tracker.draw_bboxes(output_video_frames, player_detections)
        output_video_frames = self.ball_tracker.draw_bboxes(output_video_frames, ball_detections)
        output_video_frames = self.court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
        output_video_frames = self.pose_tracker.draw_bboxes(output_video_frames, pose_detections)
        output_video_frames = mini_court.draw_mini_court(output_video_frames)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0,255,255))
        output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

        # 8. Mini Court Separate Output
        mini_court_frames = [np.zeros_like(output_video_frames[0]) for _ in output_video_frames]
        mini_court_frames = mini_court.draw_mini_court(mini_court_frames)
        mini_court_frames = mini_court.draw_points_on_mini_court(mini_court_frames, player_mini_court_detections)
        mini_court_frames = mini_court.draw_points_on_mini_court(mini_court_frames, ball_mini_court_detections, color=(0,255,255))

        for i, frame in enumerate(output_video_frames):
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if progress_callback: progress_callback("Videolar kaydediliyor...")
        save_video(output_video_frames, output_video_path, fps)
        save_video(mini_court_frames, output_mini_court_video_path, fps)

        return output_video_path, output_mini_court_video_path, player_stats_data_df

    def _calculate_stats(self, ball_shot_frames, ball_mini_court_detections, player_mini_court_detections, fps, mini_court):
        # Identify player IDs from the first frame of detections
        first_frame_dets = next((d for d in player_mini_court_detections if d), {})
        all_player_ids = sorted(list(first_frame_dets.keys()))
        
        # Map raw track IDs to 1 and 2
        # If we have < 2 players, this might crash, but choose_and_filter_players ensures 2 are chosen if available
        # But just in case, we guard against it.
        player_id_map = {}
        if len(all_player_ids) >= 2:
            player_id_map[all_player_ids[0]] = 1
            player_id_map[all_player_ids[1]] = 2
        else:
            # Fallback if detection failed seriously
            for i, pid in enumerate(all_player_ids):
                player_id_map[pid] = i + 1

        player_stats_data = [{
            'frame_num': 0,
            'player_1_number_of_shots': 0,
            'player_1_total_shot_speed': 0,
            'player_1_last_shot_speed': 0,
            'player_1_total_player_speed': 0,
            'player_1_last_player_speed': 0,
            'player_2_number_of_shots': 0,
            'player_2_total_shot_speed': 0,
            'player_2_last_shot_speed': 0,
            'player_2_total_player_speed': 0,
            'player_2_last_player_speed': 0,
        }]

        for ball_shot_ind in range(len(ball_shot_frames)-1):
            start_frame = ball_shot_frames[ball_shot_ind]
            end_frame = ball_shot_frames[ball_shot_ind+1]
            ball_shot_time_in_seconds = (end_frame-start_frame)/ fps 

            distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                            ball_mini_court_detections[end_frame][1])
            distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                            constants.DOUBLE_LINE_WIDTH,
                                                                            mini_court.get_width_of_mini_court()) 
            speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

            player_positions = player_mini_court_detections[start_frame]
            
            # Find which raw player ID shot the ball
            player_shot_ball_raw = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                    ball_mini_court_detections[start_frame][1]))
            
            # Map to 1 or 2
            player_shot_ball = player_id_map.get(player_shot_ball_raw, 1) # Default to 1 if map fail

            opponent_player_id = 1 if player_shot_ball == 2 else 2
            
            # Find raw ID for opponent to measure their speed
            # Reverse map or iterate? Just iterate find key for value
            opponent_player_id_raw_list = [k for k, v in player_id_map.items() if v == opponent_player_id]
            
            speed_of_opponent = 0
            if opponent_player_id_raw_list:
                opponent_player_id_raw = opponent_player_id_raw_list[0]
                
                # Check if opponent exists in both frames before measuring
                if (opponent_player_id_raw in player_mini_court_detections[start_frame] and 
                    opponent_player_id_raw in player_mini_court_detections[end_frame]):
                    
                    distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id_raw],
                                                                            player_mini_court_detections[end_frame][opponent_player_id_raw])
                    distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                                        constants.DOUBLE_LINE_WIDTH,
                                                                                        mini_court.get_width_of_mini_court()) 
                    speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

            current_player_stats = deepcopy(player_stats_data[-1])
            current_player_stats['frame_num'] = start_frame
            current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
            current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
            current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

            current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
            current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

            player_stats_data.append(current_player_stats)
            
        return player_stats_data

def main():
    processor = TennisMatchProcessor()
    input_video_path = "input_videos/input_video.mp4"
    
    print("Video okunuyor...")
    frames, _ = read_video(input_video_path)
    corners = None
    if frames:
        print("Köşe seçimi açılıyor...")
        corners = select_corners_manually(frames[0])
    
    processor.process_match(input_video_path, corners=corners, progress_callback=print)

if __name__ == "__main__":
    main()