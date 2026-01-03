from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import os
import sys
sys.path.append('../')
from utils import get_best_device
from utils.bbox_utils import get_center_of_bbox

class PoseTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        pose_detections = []

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                pose_detections = pickle.load(f)
            return pose_detections

        device = get_best_device()
        print(f"PoseTracker using device: {device}")

        # Batch prediction is usually faster but might consume more VRAM.
        # Let's do it similarly to PlayerTracker, loop or batch?
        # PlayerTracker loops. Let's loop to be safe and consistent.
        
        for i, frame in enumerate(frames):
            # Use TRACKING (persist=True) for better temporal consistency
            # Lower confidence to ensure we catch the far player
            results = self.model.track(frame, persist=True, conf=0.15, iou=0.5, classes=[0], device=device, verbose=False)[0]
            
            # Store keypoints: result.keypoints.xy and conf
            frame_poses = []
            if results.keypoints is not None:
                keypoints_xy = results.keypoints.xy.cpu().numpy()
                keypoints_conf = results.keypoints.conf.cpu().numpy()
                
                boxes_xyxy = results.boxes.xyxy.cpu().numpy() # We can use this for matching if needed
                
                # Zip and store
                for xy, conf, box in zip(keypoints_xy, keypoints_conf, boxes_xyxy):
                    frame_poses.append({
                        'keypoints': xy,
                        'conf': conf,
                        'bbox': box
                    })
            
            pose_detections.append(frame_poses)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(pose_detections, f)

        return pose_detections

    def draw_bboxes(self, video_frames, pose_detections, player_detections=None):
        output_video_frames = []
        
        # Skeleton config matching pose_estimation_video.py
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        KEYPOINT_CONF_THRESH = 0.3
        LINE_THICKNESS = 2

        # Helper to calculate IoU between two boxes
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0

        frame_idx = 0
        for frame, detections in zip(video_frames, pose_detections):
            vis_frame = frame 
            
            # Get player boxes for this frame if available
            player_boxes = []
            if player_detections:
                 # player_detections is a list of dicts {id: bbox} passed from main loop
                 # OR it might be just the dict for this frame if passed individually.
                 # Let's handle list input from main.py
                 if frame_idx < len(player_detections):
                     current_players = player_detections[frame_idx]
                     if isinstance(current_players, dict):
                         player_boxes = list(current_players.values())
            
            for pose in detections:
                pose_box = pose.get('bbox')
                
                # Filter: Only draw if this pose matches a tracked player
                should_draw = False
                if player_detections is None:
                    should_draw = True # Draw all if no filter provided
                elif pose_box is not None and player_boxes:
                    for p_box in player_boxes:
                        # Check IoU or containment
                        iou = calculate_iou(pose_box, p_box)
                        if iou > 0.1: # Loosely match
                            should_draw = True
                            break
                        # Check if center is inside
                        px, py = get_center_of_bbox(pose_box)
                        if p_box[0] <= px <= p_box[2] and p_box[1] <= py <= p_box[3]:
                             should_draw = True
                             break
                else: 
                     should_draw = True # Fallback if no boxes available
                
                if should_draw:
                    kpts = pose['keypoints']
                    confs = pose['conf']
                    
                    # Draw keypoints
                    for i, (x, y) in enumerate(kpts):
                        if confs[i] > KEYPOINT_CONF_THRESH:
                            cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                    
                    # Draw skeleton
                    for (start, end) in skeleton:
                        if confs[start] > KEYPOINT_CONF_THRESH and confs[end] > KEYPOINT_CONF_THRESH:
                            start_pt = (int(kpts[start][0]), int(kpts[start][1]))
                            end_pt = (int(kpts[end][0]), int(kpts[end][1]))
                            cv2.line(vis_frame, start_pt, end_pt, (255, 0, 0), LINE_THICKNESS)
            
            output_video_frames.append(vis_frame)
            frame_idx += 1
        
        return output_video_frames
