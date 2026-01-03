from easyocr import Reader
import cv2
import numpy as np

class ScoreboardTracker:
    def __init__(self, gpu=True):
        # Allow numbers only to reduce hallucinations? 
        # Tennis scores can be "15", "30", "40", "AD", "A", or sets "6", "1".
        # Initialize reader for English
        self.reader = Reader(['en'], gpu=gpu)
        self.last_score_text = ""
        self.score_history = []

    def read_score(self, frame, roi=None):
        """
        Reads text from the scoreboard ROI.
        roi: tuple (x1, y1, x2, y2) relative to frame.
        """
        if roi:
            x1, y1, x2, y2 = map(int, roi)
            # Ensure boundaries
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame

        if crop.size == 0:
            return ""

        # Preprocess? Grayscale usually helps OCR
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Simple read
        # 'allowlist' could be useful if we know format e.g. "0123456789-"
        # But names are also on scoreboard.
        results = self.reader.readtext(gray_crop, detail=0) 
        
        # Join all detected text
        full_text = " ".join(results)
        return full_text

    def process_frame(self, frame_num, frame, roi, fps):
        """
        Processes a frame, reads score, detects change.
        Returns: (has_changed, new_score_text)
        """
        # Optimize: Don't read every single frame. OCR is slow.
        # Maybe read every 1 second (fps frames)?
        # For now, we assume this is called only periodically by main loop.

        current_text = self.read_score(frame, roi)
        
        has_changed = False
        
        # Simple string comparison for now. 
        # Ideally, we parse "15-30" etc. but raw text change is a good start.
        # We need to ignore noise/flicker.
        # Maybe require the score to stay same for a few checks before accepting change?
        # For MVP, just raw change.
        
        if current_text and current_text != self.last_score_text:
            # Maybe the text is just empty or noise?
            if len(current_text) > 2: # Very basic filter
                self.last_score_text = current_text
                has_changed = True
                
                timestamp = frame_num / fps
                self.score_history.append({
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'score': current_text
                })
        
        return has_changed, current_text
