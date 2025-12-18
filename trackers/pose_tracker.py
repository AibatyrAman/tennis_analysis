from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import os
import sys
sys.path.append('../')
from utils import get_best_device

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
            # Detect
            results = self.model(frame, conf=0.25, iou=0.45, classes=[0], device=device, verbose=False)[0]
            
            # Store keypoints: result.keypoints.xy and conf
            # We need to serialize this. Tensor to numpy.
            frame_poses = []
            if results.keypoints is not None:
                keypoints_xy = results.keypoints.xy.cpu().numpy()
                keypoints_conf = results.keypoints.conf.cpu().numpy()
                
                # Zip and store
                for xy, conf in zip(keypoints_xy, keypoints_conf):
                    frame_poses.append({
                        'keypoints': xy,
                        'conf': conf
                    })
            
            pose_detections.append(frame_poses)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(pose_detections, f)

        return pose_detections

    def draw_bboxes(self, video_frames, pose_detections):
        output_video_frames = []
        
        # Skeleton config matching pose_estimation_video.py
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        KEYPOINT_CONF_THRESH = 0.5
        LINE_THICKNESS = 2

        for frame, detections in zip(video_frames, pose_detections):
            vis_frame = frame # Modify in place or copy? copy usually to be safe but main.py copies before calling.
            # main.py: output_video_frames = video_frames.copy() ... then calling draw functions sequentially.
            # So modifying 'frame' here modifies the frame in the list we are building.
            
            for pose in detections:
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
        
        return output_video_frames
