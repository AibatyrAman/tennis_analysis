from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance, get_best_device

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def choose_and_filter_players(self, court_keypoints, player_detections, corners=None):
        # 1. Define Court Boundary with Buffer for validity checks
        if corners:
            x_coords = [c[0] for c in corners]
            y_coords = [c[1] for c in corners]
            BUFFER = 80 
        else:
            x_coords = [court_keypoints[i] for i in range(0, len(court_keypoints), 2)]
            y_coords = [court_keypoints[i] for i in range(1, len(court_keypoints), 2)]
            BUFFER = 150 
            
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_y -= BUFFER; max_y += BUFFER; min_x -= BUFFER; max_x += BUFFER
        
        # Determine strict zones for fallback if needed
        mid_y = (min_y + max_y) / 2

        def is_valid_box(bbox):
            px, py = get_center_of_bbox(bbox)
            return min_x <= px <= max_x and min_y <= py <= max_y

        # Settings
        RESET_INTERVAL_FRAMES = 120 # Approx 5 seconds at 24fps. User requested 5 sec periodic check.
        filtered_player_detections = []
        
        # Current Tracking State: { Logical_ID (1 or 2): { 'detect_id': int, 'last_bbox': [x,y,x,y] } }
        tracker_state = {
            1: {'detect_id': None, 'last_bbox': None},
            2: {'detect_id': None, 'last_bbox': None}
        }

        for frame_num, frame_detections in enumerate(player_detections):
            should_reset = (frame_num % RESET_INTERVAL_FRAMES == 0)
            
            # 1. Periodic Reset / Initialization
            if should_reset:
                # Force re-detection of who is best for ID 1 (Bottom) and ID 2 (Top)
                chosen_map = self.choose_players(court_keypoints, frame_detections)
                
                # Update tracker state strictly
                for logical_id in [1, 2]:
                    det_id = chosen_map.get(logical_id)
                    if det_id is not None:
                        tracker_state[logical_id]['detect_id'] = det_id
                        tracker_state[logical_id]['last_bbox'] = frame_detections[det_id]
                    else:
                        # If lost in re-detection, keep previous state (hope for recovery) or reset?
                        # User wants "re-detect presence", if not found, we probably shouldn't track trash.
                        # But let's keep 'detect_id' as None to signal loss.
                        # However, for continuity, if we lose for 1 frame, we shouldn't flash empty.
                        # We'll rely on the recovery block below if detect_id is missing, 
                        # BUT here we mark it as None so we don't hold onto a stale OLD id if the periodic check failed.
                        # Actually, keeping the old ID might be safer if the periodical check accidentally missed a frame.
                        # But user said "every 5 seconds re-detect". So we should trust the re-detection.
                        # If re-detection returns None, it means no one is in that zone.
                        tracker_state[logical_id]['detect_id'] = None 

            # 2. Per-Frame Tracking & Output Construction
            current_frame_output = {}
            claimed_det_ids = set()
            
            # A) Try to match current tracker_state ids
            for logical_id in [1, 2]:
                det_id = tracker_state[logical_id]['detect_id']
                if det_id is not None and det_id in frame_detections:
                    # Found the tracked ID
                    bbox = frame_detections[det_id]
                    tracker_state[logical_id]['last_bbox'] = bbox
                    current_frame_output[logical_id] = bbox
                    claimed_det_ids.add(det_id)
            
            # B) Recovery for lost Logical IDs
            # If we missed a logical ID (either because it was reset to None, or the det_id vanished)
            # detect_id might be None (from reset) or just missing in frame_detections
            
            MAX_SWITCH_DISTANCE = 300
            
            for logical_id in [1, 2]:
                if logical_id in current_frame_output: continue
                
                # We need to find a new `detect_id` for this `logical_id`
                # Heuristic: Find closest available detection to 'last_bbox'
                # OR if 'last_bbox' is None (start of video), we can't do anything (wait for next reset)
                
                last_bbox = tracker_state[logical_id]['last_bbox']
                if last_bbox is None: continue
                
                last_px, last_py = get_center_of_bbox(last_bbox)
                
                best_dist = float('inf')
                best_new_det_id = None
                
                for candidate_id, candidate_bbox in frame_detections.items():
                    if candidate_id in claimed_det_ids: continue # Don't steal from the other player
                    
                    if not is_valid_box(candidate_bbox): continue
                    
                    # Strict Zone Check for Recovery too?
                    # Ideally yes, prevent P1 from grabbing a ball-boy detected at Top.
                    # P1 (Bottom) should stay Bottom.
                    cand_px, cand_py = get_center_of_bbox(candidate_bbox)
                    
                    # Optional: Add zone check here to prevent swapping lines
                    # if logical_id == 1 and cand_py < mid_y: continue # P1 must stay bottom
                    # if logical_id == 2 and cand_py > mid_y: continue # P2 must stay top
                    
                    dist = measure_distance((last_px, last_py), (cand_px, cand_py))
                    if dist < best_dist:
                        best_dist = dist
                        best_new_det_id = candidate_id
                
                if best_new_det_id is not None and best_dist < MAX_SWITCH_DISTANCE:
                    tracker_state[logical_id]['detect_id'] = best_new_det_id
                    tracker_state[logical_id]['last_bbox'] = frame_detections[best_new_det_id]
                    current_frame_output[logical_id] = frame_detections[best_new_det_id]
                    claimed_det_ids.add(best_new_det_id)
            
            filtered_player_detections.append(current_frame_output)

        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        """
        Filters and chooses players based on court zones:
        Player 1 (Bottom): Must be in the Bottom Half of the court (closer to points 2,3).
        Player 2 (Top): Must be in the Top Half of the court (closer to points 0,1).
        
        Combined with X-Axis filtering to exclude Umpires/detectors outside the lanes.
        """
        def get_center(bbox):
            return get_center_of_bbox(bbox)

        def get_distance_to_point(bbox, point):
            center = get_center(bbox)
            return measure_distance(center, point)
            
        # Determine Court Boundaries
        xs = [court_keypoints[i] for i in range(0, len(court_keypoints), 2)]
        ys = [court_keypoints[i] for i in range(1, len(court_keypoints), 2)]
        
        court_min_x = min(xs)
        court_max_x = max(xs)
        court_min_y = min(ys) 
        court_max_y = max(ys)
        
        mid_y = (court_min_y + court_max_y) / 2

        SIDE_BUFFER = 50 
        
        def is_valid_candidate(bbox, zone):
            cx, cy = get_center(bbox)
            # 1. Width Check (Exclude Umpires)
            if not ((court_min_x - SIDE_BUFFER) <= cx <= (court_max_x + SIDE_BUFFER)):
                return False
            
            # 2. Zone Check (Top vs Bottom)
            if zone == 'top':
                 return cy < mid_y + 100 
            elif zone == 'bottom':
                 return cy > mid_y - 100
                 
            return True

        # Calculate Baseline Centers
        # Top Baseline: Points 0 and 1
        top_p0 = (court_keypoints[0], court_keypoints[1])
        top_p1 = (court_keypoints[2], court_keypoints[3]) # Note: Check indices. 
        # Standard: 0,1 are x,y of pt0. 2,3 are x,y of pt1.
        # Check standard_court_drawing/detector output.
        # Usually first 4 points are corners. 
        # Let's assume indices [0,1] is P0, [2,3] is P1 etc is correct based on general usage.
        # If P0=TL, P1=TR (or similar).
        # We can just take average of ALL Top points vs ALL Bottom points?
        # Simpler: Just take (min_x, min_y) ... we already have xs, ys.
        # Top Baseline Center is roughly ( (min_x+max_x)/2, min_y ) 
        # Bottom Baseline Center is roughly ( (min_x+max_x)/2, max_y )
        # This is robust regardless of keypoint order.
        
        avg_court_x = (court_min_x + court_max_x) / 2
        top_center = (avg_court_x, court_min_y)
        bottom_center = (avg_court_x, court_max_y)

        # 1. Select Player 1 (Bottom Zone, Closest to Bottom Center)
        p1_candidates = []
        for track_id, bbox in player_dict.items():
            if not is_valid_candidate(bbox, 'bottom'): continue
            dist = get_distance_to_point(bbox, bottom_center)
            p1_candidates.append((track_id, dist))
        
        p1_candidates.sort(key=lambda x: x[1])
        player1_id = p1_candidates[0][0] if p1_candidates else None
        
        # 2. Select Player 2 (Top Zone, Closest to Top Center)
        p2_candidates = []
        for track_id, bbox in player_dict.items():
            if track_id == player1_id: continue 
            if not is_valid_candidate(bbox, 'top'): continue
            
            dist = get_distance_to_point(bbox, top_center)
            p2_candidates.append((track_id, dist))
            
        p2_candidates.sort(key=lambda x: x[1])
        player2_id = p2_candidates[0][0] if p2_candidates else None
        
        # Return explicit mapping
        return {1: player1_id, 2: player2_id}
        

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            players_dict = self.detect_frame(frame)
            player_detections.append(players_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        device = get_best_device()
        results = self.model.track(frame, persist=True, classes=[0], device=device)[0]
        
        id_name_dict = results.names
        player_dict = {}

        for box in results.boxes:
            # ID kontrolü ekledik (Bazen model ID atayamazsa kod çökmesin diye)
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                
                # Zaten classes=[0] dedik ama yapını bozmamak için bu kontrolü de tuttuk
                object_cls_id = box.cls.tolist()[0]
                object_clc_name = id_name_dict[object_cls_id]

                if object_clc_name == 'person':
                    player_dict[track_id] = result
        
        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # frame üzerine kutuları çiz
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # Kutu rengini ve yazı tipini aynen korudum
                cv2.putText(frame, f"Oyuncu ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            output_video_frames.append(frame)
        
        return output_video_frames