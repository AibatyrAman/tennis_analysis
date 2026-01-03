import cv2

def read_video(video_path):
    # WARNING: This loads all frames into memory. Use only for short videos!
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

def get_video_frames_generator(video_path):
    """
    Yields frames one by one from the video, avoiding large memory usage.
    """
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def get_video_properties(video_path):
    """
    Returns (fps, total_frames, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, width, height

def save_video(output_video_frames, output_video_path, fps):
    # YENİ: Artık dışarıdan gelen 'fps' değerini kullanıyoruz (24 yerine)
    # Note: This function still expects a list of frames. 
    # For large videos, we should use a loop outside and write directly.
    if not output_video_frames:
        return
        
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"Video kaydedildi: {output_video_path}")