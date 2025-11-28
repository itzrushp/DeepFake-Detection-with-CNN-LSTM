"""
Video Processing Utilities
"""
import cv2
import numpy as np
from typing import List

def extract_video_frames(
    video_path: str,
    max_frames: int = 30,
    fps: int = 1
) -> List[np.ndarray]:
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames = []
    frame_count = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames
