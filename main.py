from utils import (read_video, save_video)
from utils.match_processor import process_match
import os

from trackers import PlayerTracker, BallTracker


def main():
    input_video_path = "input_videos/input_video.mp4"
    filtered_video_path = "output_videos/filtered_action.mp4"
    output_video_path = "output_videos/output_video.mp4"

    if not os.path.exists("output_videos"):
        os.makedirs("output_videos")

    # Önce aksiyon sahnelerini filtrele
    print("Video işleniyor, aksiyon sahneleri ayrıştırılıyor...")
    processed_video_path = process_match(input_video_path, filtered_video_path)
    
    if processed_video_path is None:
        print("Video işleme başarısız oldu veya iptal edildi.")
        return

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


    # kutuları çizme

    # oyuncu ve top tespitlerine göre kutuları çiz
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)






    #fps değerini kaydetme fonksiyonuna gönderiyoruz
    save_video(output_video_frames, output_video_path, fps)

if __name__ == "__main__":
    main()