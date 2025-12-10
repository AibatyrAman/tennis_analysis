import cv2
import numpy as np

class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 550
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] = int(self.court_start_x)
        drawing_key_points[1] = int(self.court_start_y)
        # point 1
        drawing_key_points[2] = int(self.court_end_x)
        drawing_key_points[3] = int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = int(self.court_start_y + 14.4*10) # imaginary service line
        # point 3
        drawing_key_points[6] = int(self.court_start_x)
        drawing_key_points[7] = int(self.court_end_y)
        # point 4
        drawing_key_points[8] = int(self.court_end_x)
        drawing_key_points[9] = int(self.court_end_y)
        # point 5
        drawing_key_points[10] = int(self.court_start_x + 60) # imaginary double alley
        drawing_key_points[11] = int(self.court_start_y)
        # point 6 - this is wrong in typical hardcoded values but keep structure
        # ... Let's use proportional implementation instead of hardcoded keypoints logic 
        # to ensure it matches 8.23m x 23.77m accurately
        pass 
    
    def normalize_to_mini_court(self, point_meters):
        # Point is (x, y) in meters where (0,0) is top-left of the court
        # Court dimensions: width=10.97m, length=23.77m
        
        # Scaling factors
        court_width_meters = 10.97
        court_length_meters = 23.77
        
        scale_x = self.court_drawing_width / court_width_meters
        scale_y = self.court_drawing_height / court_length_meters
        
        x_pixel = self.court_start_x + (point_meters[0] * scale_x)
        y_pixel = self.court_start_y + (point_meters[1] * scale_y)
        
        return (int(x_pixel), int(y_pixel))

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = frame.copy()
            
            # Draw background
            cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
            cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 0), 2)
            
            # Draw Court Lines
            # Outer boundary
            cv2.rectangle(frame, (self.court_start_x, self.court_start_y), (self.court_end_x, self.court_end_y), (0, 0, 0), 2)
            
            # Net (Center)
            net_y = int(self.court_start_y + self.court_drawing_height / 2)
            cv2.line(frame, (self.court_start_x, net_y), (self.court_end_x, net_y), (255, 0, 0), 2)
            
            # Half Court Line (Vertical Center) - only between service lines
            center_x = int(self.court_start_x + self.court_drawing_width / 2)
            service_line_top_y = int(self.court_start_y + self.court_drawing_height * (5.5/23.77)) # Approx 5.5m from end
            service_line_bottom_y = int(self.court_end_y - self.court_drawing_height * (5.5/23.77))
            
            # Service Lines
            # Top Service Line
            # Service boxes are 6.4m from the net
            # So 11.885 - 6.4 = 5.485m from baseline
            
            dist_from_baseline_param = 5.485 / 23.77
            service_y_top = int(self.court_start_y + self.court_drawing_height * dist_from_baseline_param)
            service_y_bottom = int(self.court_end_y - self.court_drawing_height * dist_from_baseline_param)
            
            cv2.line(frame, (self.court_start_x, service_y_top), (self.court_end_x, service_y_top), (0, 0, 0), 2)
            cv2.line(frame, (self.court_start_x, service_y_bottom), (self.court_end_x, service_y_bottom), (0, 0, 0), 2)
            
            # Center Service Line
            cv2.line(frame, (center_x, service_y_top), (center_x, service_y_bottom), (0, 0, 0), 2)
            
            # Singles Sidelines
            # 1.37m from doubles sideline
            single_margin_param = 1.37 / 10.97
            single_x_left = int(self.court_start_x + self.court_drawing_width * single_margin_param)
            single_x_right = int(self.court_end_x - self.court_drawing_width * single_margin_param)
            
            cv2.line(frame, (single_x_left, self.court_start_y), (single_x_left, self.court_end_y), (0, 0, 0), 2)
            cv2.line(frame, (single_x_right, self.court_start_y), (single_x_right, self.court_end_y), (0, 0, 0), 2)
            
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_height_of_mini_court(self):
        return self.court_drawing_height
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2), (4, 5), (6,7), (1,3), 
            (0,1), (8,9), (10,11), (10,11)
            ]

    def get_mini_court_coordinates(self, object_position, corners):
        """
        Görüntü üzerindeki pozisyonu mini kort koordinatlarına çevirir.
        
        Args:
            object_position (tuple): (x, y) coordinates on the screen.
            corners (np.array): 4 corners of the court in the image (screen coordinates).
                                Order: top-left, top-right, bottom-right, bottom-left
        """
        corners = corners.astype(np.float32)
        
        # Real world dimensions (meters)
        # Using the standard tennis court dimensions
        # Width: 10.97m
        # Length: 23.77m
        court_width = 10.97
        court_length = 23.77
        
        # Destination points in real world (meters)
        # We map image to (0,0) -> (width, length) space
        dst_points = np.float32([
            [0, 0],              # Top-Left
            [court_width, 0],    # Top-Right
            [court_width, court_length], # Bottom-Right
            [0, court_length]    # Bottom-Left
        ])
        
        # Get Homography Matrix using the detected corners
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Transform the object position
        point = np.array([object_position], dtype=np.float32)
        point = np.array([point]) # Needs shape (1, 1, 2)
        
        transformed_point = cv2.perspectiveTransform(point, matrix)[0][0]
        
        # Transformed point is in meters relative to top-left court corner
        
        # Now convert meters to mini-court pixels
        mini_court_point = self.normalize_to_mini_court(transformed_point)
        
        return mini_court_point

    def draw_heatmap(self, frames, bounce_points, corners):
        """
        Her karede mini kortu ve tüm sekme noktalarını içeren ısı haritası benzeri noktaları çizer.
        """
        output_frames = []
        
        # Convert all bounce points to mini court coordinates first
        mini_court_bounces = []
        for point in bounce_points:
            mc_point = self.get_mini_court_coordinates(point, corners)
            mini_court_bounces.append(mc_point)
            
        for frame in frames:
            # First draw the mini court board
            # In draw_mini_court logic, the background is redrawn every time
            # We can reuse draw_mini_court but it processes list
            # Let's simplify and draw directly here since we usually want to overlay tracking AND heatmap
            
            # Assuming 'draw_mini_court' was already called or we call it per frame
            # Let's assume we modify the frame directly
            
            # Background
            cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
            cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 0), 2)
            
            # Draw empty court first
            dummy_frames = self.draw_mini_court([frame])
            frame = dummy_frames[0]
            
            # Draw Bounce Points (Heatmap style)
            # We will draw semi-transparent circles
            overlay = frame.copy()
            for point in mini_court_bounces:
                # Check if point is inside drawing area (roughly)
                if (self.start_x < point[0] < self.end_x) and (self.start_y < point[1] < self.end_y):
                    cv2.circle(overlay, point, 10, (0, 0, 255), -1) # Red dots for bounces
            
            # Apply transparency
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            output_frames.append(frame)
            
        return output_frames
