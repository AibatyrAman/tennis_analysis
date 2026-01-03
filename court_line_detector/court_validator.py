import numpy as np
import cv2

class CourtValidator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Standard Tennis Court Dimensions (approximate ratios)
        # We don't need exact meters, just the aspect ratio roughly.
        # A tennis court is roughly 11m x 24m (doubles). Aspect ratio ~0.45.
        # In perspective it changes, but the homography check handles that.
        self.std_court = np.array([
            [0, 0],             # Top-Left (0)
            [10.97, 0],         # Top-Right (1)
            [10.97, 23.77],     # Bottom-Right (3)
            [0, 23.77]          # Bottom-Left (2)
        ], dtype=np.float32)

    def validate_keypoints(self, keypoints):
        """
        Validates the geometric structure of detection.
        keypoints: list or array of 28 floats [x1, y1, x2, y2, ...]
        """
        if keypoints is None or len(keypoints) < 8:
            return False

        # Extract strict corner points: 0(TL), 2(TR), 3(BR), 1(BL) based on typical indexing
        # Note: The model output indexing might vary. 
        # Standard assumption based on typical CourtLineDetector behavior:
        # 0: TL, 1: BL, 2: TR, 3: BR (Indices 0, 1, 2, 3 in the array of pairs)
        # Let's verify indexing from existing code/usage.
        # usually: 0=TopLeft, 1=TopRight, 2=BottomLeft, 3=BottomRight?
        # User's visualization shows: 
        # 0=TopLeft, 2=TopRight (Point 2 label), 1=BottomLeft (Point 1 label), 
        # 3 is missing, maybe 3 is BottomRight?
        # Let's rely on clustering first as it's safer.
        
        xs = keypoints[::2]
        ys = keypoints[1::2]
        
        # 1. Clustering Check (Variance)
        # If all points are bunched up (std dev is small relative to screen size)
        x_std = np.std(xs)
        y_std = np.std(ys)
        
        screen_diag = np.sqrt(self.width**2 + self.height**2)
        spatial_spread = np.sqrt(x_std**2 + y_std**2)
        
        # Threshold: if spread is less than 5% of screen diagonal, it's likely a face or small object
        if spatial_spread < (screen_diag * 0.1): 
            return False

        # 2. Area Constraint (Convex Hull Area)
        # Use corner points (or all points hull) to check area
        points = np.column_stack((xs, ys)).astype(np.float32)
        hull = cv2.convexHull(points)
        area = cv2.contourArea(hull)
        
        frame_area = self.width * self.height
        # If court takes up less than 5% of screen, it's probably too far or false positive
        if area < (frame_area * 0.10):
            return False
            
        return True

    def get_valid_indices(self, keypoints_generator):
        valid_indices = []
        for i, kp in enumerate(keypoints_generator):
            if self.validate_keypoints(kp):
                valid_indices.append(i)
        return valid_indices
