"""
Face Detection and Extraction using MTCNN
"""
from mtcnn import MTCNN
import numpy as np
from typing import List, Dict

class FaceExtractor:
    """Face detection using MTCNN"""

    def __init__(self, min_confidence: float = 0.9):
        self.detector = MTCNN()
        self.min_confidence = min_confidence

    def extract(self, image: np.ndarray) -> List[Dict]:
        """Extract faces from image"""
        detections = self.detector.detect_faces(image)

        faces = []

        for detection in detections:
            confidence = detection['confidence']

            if confidence < self.min_confidence:
                continue

            x, y, w, h = detection['box']
            x = max(0, x)
            y = max(0, y)

            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)

            face = image[y1:y2, x1:x2]

            faces.append({
                'face': face,
                'box': (x, y, w, h),
                'confidence': confidence
            })

        faces.sort(key=lambda x: x['confidence'], reverse=True)
        return faces

_extractor = None

def extract_faces(image: np.ndarray) -> List[Dict]:
    """Helper function"""
    global _extractor

    if _extractor is None:
        _extractor = FaceExtractor()

    return _extractor.extract(image)
