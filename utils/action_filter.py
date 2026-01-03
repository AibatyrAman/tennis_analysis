import cv2
import numpy as np

class ActionFilter:
    def __init__(self, reference_frame, roi_corners=None, similarity_threshold=0.85):
        """
        Initializes the action filter with a reference frame and optionally a specific ROI.
        """
        self.roi_mask = None
        
        if roi_corners and len(roi_corners) == 4:
            # Create a mask for the ROI
            h, w = reference_frame.shape[:2]
            self.roi_mask = np.zeros((h, w), dtype=np.uint8)
            points = np.array([roi_corners], dtype=np.int32)
            cv2.fillPoly(self.roi_mask, points, 255)
            
        self.reference_frame = self._preprocess_frame(reference_frame)
        self.similarity_threshold = similarity_threshold

    def _preprocess_frame(self, frame):
        """
        Converts to grayscale, applies ROI if set, and resizes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.roi_mask is not None:
            # Access bitwise_and directly through cv2 (safest) or just use the mask
            gray = cv2.bitwise_and(gray, gray, mask=self.roi_mask)
            
            # Optional: crop to bounding rect of mask to remove black borders
            # x, y, w, h = cv2.boundingRect(self.roi_points)
            # gray = gray[y:y+h, x:x+w]
            
        # Resize to a small fixed size to be insensitive to small pixel noises but sensitive to global scene changes
        resized = cv2.resize(gray, (128, 72)) 
        return resized

    def calculate_similarity(self, frame):
        """
        Calculates structural similarity (using correlation) between the frame and the reference.
        Returns a score between 0 and 1.
        """
        processed_frame = self._preprocess_frame(frame)
        
        # Use template matching as a similarity metric (Normalized Correlation)
        res = cv2.matchTemplate(processed_frame, self.reference_frame, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        return score

    def filter_frames(self, frames):
        """
        Filters the list of frames, keeping only those that are similar to the reference.
        Returns the filtered list of frames.
        """
        print("Filtering frames based on action/court visibility...")
        filtered_frames = []
        original_count = len(frames)
        
        for i, frame in enumerate(frames):
            score = self.calculate_similarity(frame)
            if score > self.similarity_threshold:
                filtered_frames.append(frame)
            
            if i % 100 == 0:
                print(f"Checking frame {i}/{original_count}, Similarity: {score:.2f}")

        print(f"Filtering complete. Kept {len(filtered_frames)}/{original_count} frames.")
        return filtered_frames
        
    def get_filtered_indices(self, frames_generator, total_frames=None):
        """
        Iterates through frames from a generator and returns a boolean list or set of indices 
        that passed the filter. Memory efficient.
        """
        print("Analyzing frames for action filtering...")
        valid_indices = set()
        
        for i, frame in enumerate(frames_generator):
            score = self.calculate_similarity(frame)
            if score > self.similarity_threshold:
                valid_indices.add(i)
                
            if i % 200 == 0:
                prog = f"{i}" if total_frames is None else f"{i}/{total_frames}"
                print(f"Action Filter: Frame {prog}, Similarity: {score:.2f}")
                
        return valid_indices
