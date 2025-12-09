import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # YENİ: Videonun orijinal FPS değerini öğreniyoruz
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Video okundu. Kare: {len(frames)}, FPS: {fps}")
    
    # ARTIK İKİ ŞEY DÖNDÜRÜYORUZ: Kareler ve FPS
    return frames, fps

def save_video(output_video_frames, output_video_path, fps):
    # YENİ: Artık dışarıdan gelen 'fps' değerini kullanıyoruz (24 yerine)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"Video kaydedildi: {output_video_path}")