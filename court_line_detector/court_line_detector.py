import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import get_best_device

class CourtLineDetector:
    def __init__(self, model_path):
        self.device = get_best_device()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # Image is typically NumPy array (H, W, C) BGR
        # Convert to RGB for model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
        keypoints = outputs.cpu().numpy().flatten()
        
        # Scale keypoints back to original image size
        # Model output is in 224x224 space
        original_h, original_w = image.shape[:2]
        
        # Keypoints are [x1, y1, x2, y2, ...]
        keypoints[::2] *= (original_w / 224.0)
        keypoints[1::2] *= (original_h / 224.0)

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            # Skip points if logic suggests but typically these are always predicted
            
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames):
        output_video_frames = []
        for frame in video_frames:
            keypoints = self.predict(frame)
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames