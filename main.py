import cv2
import os
import sys
import shutil
import pandas as pd
import numpy as np
from copy import deepcopy

# Utils
from utils import (
    read_video, 
    save_video, 
    measure_distance, 
    convert_pixel_distance_to_meters,
    get_video_properties, 
    get_video_frames_generator
)
from utils.player_stats_drawer_utils import draw_player_stats
import constants

# Trackers & Detectors
from trackers import PlayerTracker, BallTracker, PoseTracker, ScoreboardTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from utils.action_filter import ActionFilter
from court_line_detector.court_validator import CourtValidator

class TennisMatchProcessor:
    def __init__(self, 
                 model_path_yolo='models/yolov8x.pt', 
                 model_path_ball='models/yeni_model.pt', 
                 model_path_court='models/keypoints_model_50.pt', 
                 model_path_pose='models/yolo11x-pose.pt'):
        
        self.player_tracker = PlayerTracker(model_path=model_path_yolo)
        self.ball_tracker = BallTracker(model_path=model_path_ball)
        self.pose_tracker = PoseTracker(model_path=model_path_pose)
        self.court_line_detector = CourtLineDetector(model_path_court)
        self.scoreboard_tracker = ScoreboardTracker(gpu=True)
        self.court_validator = None # Initialized after dimensions are known

    def process_match(self, input_video_path, corners=None, scoreboard_roi=None, progress_callback=None):
        """
        Main pipeline function.
        corners: Argument kept for compatibility but ignored in logic (Auto detection used).
        """
        
        # --- 0. Setup Outputs ---
        if not os.path.exists("output_videos"):
            os.makedirs("output_videos")
            
        base_name = os.path.basename(input_video_path)
        output_dir = os.path.abspath("output_videos")
        output_video_path = os.path.join(output_dir, f"output_{base_name}")
        output_mini_court_video_path = os.path.join(output_dir, f"mini_court_{base_name}")

        # --- 1. Video Info ---
        if progress_callback: progress_callback("Video bilgileri alınıyor...")
        fps, total_frames, width, height = get_video_properties(input_video_path)
        
        # Initialize Validator
        self.court_validator = CourtValidator(width, height)
        
        # Get a reference frame (e.g. Frame 50) for static logic (MiniCourt, ActionFilter)
        # Choosing a frame slightly into the video avoids potential black/transition frames at start.
        target_frame_idx = min(50, total_frames - 1) 
        reference_frame = None
        
        # We need to iterate to reach frame 50
        gen = get_video_frames_generator(input_video_path)
        for i, f in enumerate(gen):
            if i == target_frame_idx:
                reference_frame = f
                break
            # Keep first frame as fallback
            if i == 0 and reference_frame is None:
                reference_frame = f
                
        if reference_frame is None:
            print("Hata: Video okunamadı veya çok kısa.")
            return None, None, None, None

        # --- 2. Auto Court Detection (Reference) ---
        if progress_callback: progress_callback("Referans kort çizgileri tespit ediliyor...")
        if progress_callback: progress_callback("Referans kort çizgileri tespit ediliyor...")
        # Static keypoints for Logic (ActionFilter, MiniCourt initialization)
        court_keypoints = self.court_line_detector.predict(reference_frame)

        # Build Scaled Court Polygon for Ball Filter
        xs = court_keypoints[::2]
        ys = court_keypoints[1::2]
        points = np.column_stack((xs, ys)).astype(np.int32)
        hull = cv2.convexHull(points)

        def scale_polygon(polygon, scale_x=1.1, scale_y=1.3):
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
        
        # Apply strict buffer based on user feedback
        scaled_court_polygon = scale_polygon(hull, scale_x=1.1, scale_y=1.3)
        self.ball_tracker.set_court_polygon(scaled_court_polygon)

        # --- 3. Action Filtering ---
        # Define ROI using the bounding box of ALL detected keypoints.
        # This is safer than relying on specific indices (0,4,6,2) which might be wrong.
        all_x = court_keypoints[::2]
        all_y = court_keypoints[1::2]
        roi_min_x = max(0, int(min(all_x)))
        roi_min_y = max(0, int(min(all_y)))
        roi_max_x = min(width, int(max(all_x)))
        roi_max_y = min(height, int(max(all_y)))
        
        # Expand ROI slightly with buffer
        buffer = 50
        auto_roi = [
            (max(0, roi_min_x - buffer), max(0, roi_min_y - buffer)), # TL
            (min(width, roi_max_x + buffer), max(0, roi_min_y - buffer)), # TR
            (min(width, roi_max_x + buffer), min(height, roi_max_y + buffer)), # BR
            (max(0, roi_min_x - buffer), min(height, roi_max_y + buffer))  # BL
        ]
        
        if progress_callback: progress_callback("Aksiyon harici kareler filtreleniyor...")
        
        action_filter = ActionFilter(reference_frame, roi_corners=auto_roi, similarity_threshold=0.6)
        video_gen_filter = get_video_frames_generator(input_video_path)
        action_valid_indices = action_filter.get_filtered_indices(video_gen_filter, total_frames=total_frames)
        
        # --- 3.5 Geometric Validation ---
        if progress_callback: progress_callback("Geometrik Doğrulama yapılıyor...")
        
        final_valid_indices = []
        # Pre-calculated court keypoints for valid frames to avoid re-running in draw loop
        # Map: frame_idx -> keypoints
        self.frame_court_keypoints = {} 
        
        video_gen_validator = get_video_frames_generator(input_video_path)
        action_indices_set = set(action_valid_indices)
        
        for i, frame in enumerate(video_gen_validator):
            if i in action_indices_set:
                # Run detector
                kps = self.court_line_detector.predict(frame)
                if self.court_validator.validate_keypoints(kps):
                    final_valid_indices.append(i)
                    self.frame_court_keypoints[i] = kps
        
        valid_indices = final_valid_indices

        if not valid_indices:
            print("Uyarı: Hiçbir kare seçilmedi. Tüm video işlenemez.")
            return None, None, None, None
            
        print(f"Saklanan kare sayısı (Geometrik Filtre Sonrası): {len(valid_indices)} / {total_frames}")

        # --- 4. Tracking (Batch/Generator) ---
        # We process ALL frames to ensure tracking continuity (persist=True relies on history)
        
        if progress_callback: progress_callback("Oyuncular takip ediliyor...")
        video_gen_player = get_video_frames_generator(input_video_path)
        player_detections = self.player_tracker.detect_frames(
            video_gen_player, 
            read_from_stub=False, 
            stub_path="tracker_stubs/player_detections.pkl"
        )
        
        if progress_callback: progress_callback("Top takip ediliyor (Kort Filtreli + Interpolasyon)...")
        video_gen_ball = get_video_frames_generator(input_video_path)
        ball_detections = self.ball_tracker.detect_frames(video_gen_ball, read_from_stub=False)
        ball_detections = self.ball_tracker.interpolate_ball_positions(ball_detections)

        if progress_callback: progress_callback("İskelet takibi yapılıyor...")
        video_gen_pose = get_video_frames_generator(input_video_path)
        pose_detections = self.pose_tracker.detect_frames(
            video_gen_pose, 
            read_from_stub=False, 
            stub_path="tracker_stubs/pose_detections.pkl"
        )

        # --- 5. Player Filtering & Recovery ---
        # Using the robust logic verified in tests/test_player_recovery.py
        # Passing None for corners forces auto-detected court keypoints usage
        player_detections = self.player_tracker.choose_and_filter_players(
            court_keypoints, 
            player_detections, 
            corners=None 
        )

        # --- 6. Mini Court & Stats Calculation ---
        if progress_callback: progress_callback("İstatistikler hesaplanıyor...")
        
        mini_court = MiniCourt(reference_frame) 
        
        # 6.1 Detect Ball Bounces (Ground Hits)
        # Returns list of [frame_num, center_x, center_y]
        ball_bounce_frames = self.ball_tracker.get_ball_bounce_frames(ball_detections)
        print(f"DEBUG: Detected {len(ball_bounce_frames)} ball bounces.")
        
        # 6.2 Scoreboard Analysis & Heatmap
        if progress_callback: progress_callback("Skor ve ısı haritası hazırlıkları...")
        scoreboard_tracker = self.scoreboard_tracker # Use shared instance
        
        # Scoreboard ROI: Assuming top 20% of screen to save simple OCR time/noise
        # If scoreboard is at bottom, this needs adjustment. 
        # Standard broadcast is usually top-left or bottom-left.
        # Let's use whole top strip for now.
        scoreboard_roi = (0, 0, width, int(height * 0.2)) 
        
        # Store accumulated winning points for heatmap
        winning_bounce_positions = []
        
        ball_shot_frames = self.ball_tracker.get_ball_shot_frames(ball_detections)
        
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )
        
        player_stats_data = self._calculate_stats(
            ball_shot_frames, 
            ball_mini_court_detections, 
            player_mini_court_detections, 
            fps, 
            mini_court
        )

        # Prepare Metrics DataFrame
        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({'frame_num': list(range(total_frames))})
        player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
        player_stats_data_df = player_stats_data_df.ffill()

        # Calculate Averages
        for p in [1, 2]:
            shots = player_stats_data_df[f'player_{p}_number_of_shots']
            player_stats_data_df[f'player_{p}_average_shot_speed'] = player_stats_data_df[f'player_{p}_total_shot_speed'] / shots.replace(0, 1)
            opponent = 2 if p == 1 else 1
            opp_shots = player_stats_data_df[f'player_{opponent}_number_of_shots']
            player_stats_data_df[f'player_{p}_average_player_speed'] = player_stats_data_df[f'player_{p}_total_player_speed'] / opp_shots.replace(0, 1)

        # --- 7. Visualization ---
        if progress_callback: progress_callback("Görselleştirme yapılıyor...")
        # Get frames again for drawing
        video_gen_draw = get_video_frames_generator(input_video_path)
        
        # Initialize lists to store processed frames
        output_video_frames = []
        mini_court_frames = []
        
        # Perfect Mini Court for display
        mini_court_width = 350
        mini_court_height = 600
        dummy_frame = np.zeros((mini_court_height, mini_court_width, 3), dtype=np.uint8)
        perfect_mini_court = MiniCourt(dummy_frame)
        
        # Convert detections for the View (Perfect/Clean Mini Court)
        p_mini_court_detections_view, b_mini_court_detections_view = perfect_mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )

        # Writers
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_main = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        out_mini = cv2.VideoWriter(output_mini_court_video_path, fourcc, fps, (mini_court_width, mini_court_height))
        
        score_events = [] 
        
        # Determine strict ROI for Scoreboard if provided
        real_scoreboard_roi = None
        if scoreboard_roi:
            # ROI is normalized (0-1), scale to pixels
            sx1, sy1, sx2, sy2 = scoreboard_roi
            # If roi comes from app.py it might be already scaled or normalized. 
            # Assuming pixel coords if coming from manual selector or similar, 
            # BUT app.py usually passes normalized 0-1 values if it's new code.
            # Let's verify input. 
            # If coordinates are small floats < 1, scale them.
            if sx1 <= 1 and sy1 <= 1 and sx2 <= 1 and sx2 <= 1:
                 sx1, sy1, sx2, sy2 = int(sx1*width), int(sy1*height), int(sx2*width), int(sy2*height)
            real_scoreboard_roi = (int(sx1), int(sy1), int(sx2), int(sy2))

        for i, frame in enumerate(video_gen_draw):
            if i not in valid_indices:
                continue
                
            # -- Draw Pipeline --
            
            # 1. Main Objects
            # Helper functions usually expect a list of frames
            frame = self.player_tracker.draw_bboxes([frame], [player_detections[i]])[0]
            frame = self.ball_tracker.draw_bboxes([frame], [ball_detections[i]])[0]
            
            # 2. Court Lines - Dynamic Drawing
            # Use pre-calculated validated keypoints (from step 3.5)
            # If for some reason key is missing (shouldn't happen for valid_indices), predict fallback
            current_court_keypoints = self.frame_court_keypoints.get(i)
            if current_court_keypoints is None:
                 current_court_keypoints = self.court_line_detector.predict(frame)
            frame = self.court_line_detector.draw_keypoints(frame, current_court_keypoints)

            # 3. Pose/Skeleton
            if i < len(pose_detections):
                frame = self.pose_tracker.draw_bboxes([frame], [pose_detections[i]], player_detections=[player_detections[i]])[0]
            
            
            # 4. Stats Overlay
            if i < len(player_stats_data_df):
                frame = draw_player_stats([frame], player_stats_data_df.iloc[[i]])[0]
                
            # 5. Scoreboard & Heatmap Update
            if real_scoreboard_roi and i % int(fps) == 0:
                has_changed, current_score = scoreboard_tracker.process_frame(i, frame, real_scoreboard_roi, fps)
                if has_changed:
                    print(f"DEBUG: Score changed at frame {i}: {current_score}")
                    score_events.append({'frame': i, 'timestamp': i/fps, 'score': current_score})
                    
                    # HEATMAP LOGIC: Find last bounce
                    # Lookback exactly 5 seconds
                    lookback_frames = int(5 * fps)
                    search_start = max(0, i - lookback_frames)
                    
                    candidates = [b for b in ball_bounce_frames if search_start < b[0] < i]
                    print(f"DEBUG: Search window [{search_start}, {i}] (5s). Candidates: {len(candidates)}")
                    
                    if candidates:
                        last_bounce = candidates[-1] # [frame, x, y]
                        bounce_point = (last_bounce[1], last_bounce[2])
                        start_frame_idx = last_bounce[0]
                        kps = self.frame_court_keypoints.get(int(start_frame_idx), court_keypoints)
                        
                        # Use perfect_mini_court to map to 350x600 coordinate space
                        mc_point = perfect_mini_court.get_mini_court_coordinates_from_point(bounce_point, kps)
                        if mc_point:
                            winning_bounce_positions.append({
                                'pos': mc_point,
                                'frame': start_frame_idx,
                                'timestamp': start_frame_idx/fps,
                                'score': current_score
                            })
                
                # Visual Feedback for scoreboard
                sx1, sy1, sx2, sy2 = real_scoreboard_roi
                cv2.putText(frame, f"Score: {current_score}", (sx1, sy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)

            # 6. Metadata
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write Main
            out_main.write(frame)
            
            # Write Mini
            mini_frame = np.zeros((mini_court_height, mini_court_width, 3), dtype=np.uint8)
            mini_frame = perfect_mini_court.draw_mini_court([mini_frame])[0]
            
            # Draw Heatmap (Accumulated Bounces)
            # Use stored detailed events
            # Draw Heatmap (Accumulated Bounces)
            # Use stored detailed events
            for idx, event in enumerate(winning_bounce_positions):
                pt = event['pos']
                
                # Fading Logic:
                # Calculate intensity based on index (0 to 1)
                # Newer events (higher index) -> Brighter red
                # Older events (lower index) -> Darker red
                
                total_events = len(winning_bounce_positions)
                # Prevent division by zero
                ratio = (idx + 1) / total_events if total_events > 0 else 1
                
                # Min brightness 100, Max 255
                intensity = int(100 + (155 * ratio))
                
                # Color (BGR): Blue=0, Green=0, Red=intensity
                color = (0, 0, intensity)
                
                # Make the very last one distinct (e.g. larger or different color) if desired
                # But user asked for "shades".
                radius = 7
                if idx == total_events - 1:
                    radius = 10 # Highlight last one slightly bigger
                    color = (0, 0, 255) # Pure bright red
                
                cv2.circle(mini_frame, (int(pt[0]), int(pt[1])), radius, color, -1)
            
            if i < len(p_mini_court_detections_view):
                mini_frame = perfect_mini_court.draw_points_on_mini_court([mini_frame], [p_mini_court_detections_view[i]])[0]
            if i < len(b_mini_court_detections_view):
                mini_frame = perfect_mini_court.draw_points_on_mini_court([mini_frame], [b_mini_court_detections_view[i]], color=(0,255,255))[0]
            out_mini.write(mini_frame)

        out_main.release()
        out_mini.release()
        
        print(f"Video kaydedildi: {output_video_path}")
        return output_video_path, output_mini_court_video_path, player_stats_data_df, winning_bounce_positions

    def _calculate_stats(self, ball_shot_frames, ball_mini_court_detections, player_mini_court_detections, fps, mini_court):
        # Identify player IDs safety
        first_frame_dets = {}
        for dets in player_mini_court_detections:
            if dets:
                first_frame_dets = dets
                break
        
        all_player_ids = sorted(list(first_frame_dets.keys()))
        player_id_map = {}
        
        # Map tracking IDs to Player 1 / Player 2
        # Usually checking initial x-position (left=1, right=2) or just order
        # For simplicity, just mapping sorted IDs
        if len(all_player_ids) >= 2:
            player_id_map[all_player_ids[0]] = 1
            player_id_map[all_player_ids[1]] = 2
        else:
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
            
            if ball_shot_time_in_seconds == 0: continue
            if start_frame >= len(ball_mini_court_detections) or end_frame >= len(ball_mini_court_detections): continue
            
            # Ball must exist
            if 1 not in ball_mini_court_detections[start_frame] or 1 not in ball_mini_court_detections[end_frame]: continue
            if not player_mini_court_detections[start_frame]: continue

            # Speed Calc
            distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                            ball_mini_court_detections[end_frame][1])
            distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                            constants.DOUBLE_LINE_WIDTH,
                                                                            mini_court.get_width_of_mini_court()) 
            speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

            # Identify Shooter
            player_positions = player_mini_court_detections[start_frame]
            player_shot_ball_raw = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                    ball_mini_court_detections[start_frame][1]))
            player_shot_ball = player_id_map.get(player_shot_ball_raw, 1) 

            # Opponent Speed
            opponent_player_id = 1 if player_shot_ball == 2 else 2
            opponent_player_id_raw_list = [k for k, v in player_id_map.items() if v == opponent_player_id]
            
            speed_of_opponent = 0
            if opponent_player_id_raw_list:
                opponent_player_id_raw = opponent_player_id_raw_list[0]
                if (opponent_player_id_raw in player_mini_court_detections[start_frame] and 
                    opponent_player_id_raw in player_mini_court_detections[end_frame]):
                    
                    dist_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id_raw],
                                                   player_mini_court_detections[end_frame][opponent_player_id_raw])
                    dist_meters = convert_pixel_distance_to_meters(dist_pixels,
                                                                constants.DOUBLE_LINE_WIDTH,
                                                                mini_court.get_width_of_mini_court()) 
                    speed_of_opponent = dist_meters/ball_shot_time_in_seconds * 3.6

            # Update Stats
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
    print("Main Started...")
    processor = TennisMatchProcessor()
    # Default behavior for standalone run
    input_video_path = "input_videos/input_video02_01.mp4"
    if len(sys.argv) > 1:
        input_video_path = sys.argv[1]
        
    if os.path.exists(input_video_path):
        processor.process_match(input_video_path, progress_callback=print)
    else:
        print(f"Video bulunamadı: {input_video_path}")

if __name__ == "__main__":
    main()