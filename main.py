from utils import (read_video, save_video)
from utils.match_processor import process_match
from utils.mini_court import MiniCourt
import os
import pandas as pd
import cv2

from trackers import PlayerTracker, BallTracker


def main():
    input_video_path = "input_videos/input_video.mp4"
    filtered_video_path = "output_videos/filtered_action.mp4"
    output_video_path = "output_videos/output_video.mp4"

    if not os.path.exists("output_videos"):
        os.makedirs("output_videos")

    # Önce aksiyon sahnelerini filtrele
    print("Video işleniyor, aksiyon sahneleri ayrıştırılıyor...")
    result = process_match(input_video_path, filtered_video_path)
    
    if result is None:
        print("Video işleme başarısız oldu veya iptal edildi.")
        return
        
    processed_video_path, corners = result

    # hem kareleri hem de fps'i alıyoruz (Artık filtrelenmiş videoyu okuyoruz)
    video_frames, fps = read_video(processed_video_path)

    # oyuncu ve top takibi
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/updated_new_best.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                    read_from_stub=False,
                                                    stub_path="tracker_stubs/player_detections.pkl")
                                                    
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                read_from_stub=False,
                                                stub_path="tracker_stubs/ball_detections.pkl")
    
    # Top pozisyonlarını interpolate et (eksik verileri tamamla)
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # Sekme anlarını tespit et
    bounce_frame_indices = ball_tracker.get_ball_shot_frames(ball_detections)
    
    bounce_points = []
    for frame_idx in bounce_frame_indices:
        ball_dict = ball_detections[frame_idx]
        if 1 in ball_dict:
            bbox = ball_dict[1]
            # Topun merkezi (veya alt noktası)
            # x_center = (bbox[0] + bbox[2]) / 2
            # y_center = (bbox[1] + bbox[3]) / 2
            # Genellikle topun alt noktası yere temas eder
            x_center = (bbox[0] + bbox[2])/2
            y_bottom = bbox[3] 
            bounce_points.append((x_center, y_bottom))


    # kutuları çizme

    # oyuncu ve top tespitlerine göre kutuları çiz
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    # Mini kort ve ısı haritasını çiz
    # İlk kareden mini kort için referans alabiliriz, ama draw_heatmap her kareye uygulayacak
    if len(video_frames) > 0:
        mini_court = MiniCourt(video_frames[0])
        output_video_frames = mini_court.draw_heatmap(output_video_frames, bounce_points, corners)

    #fps değerini kaydetme fonksiyonuna gönderiyoruz
    save_video(output_video_frames, output_video_path, fps)

if __name__ == "__main__":
    main()