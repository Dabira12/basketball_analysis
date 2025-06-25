import cv2
import numpy as np
from paddleocr import PaddleOCR

class JerseyNumberDetector:
    def __init__(self):
        # Initialize PaddleOCR with minimal configuration
        self.reader = PaddleOCR(
            lang='en',
            det_model_dir='ch_PP-OCRv3_det_infer',  # Use default detection model
            rec_model_dir='ch_PP-OCRv3_rec_infer',  # Use default recognition model
            use_angle_cls=True,
            use_gpu=False
        )
        
    def get_jersey_number(self, frame, player_bbox):
        """Extract jersey number from a player's bounding box"""
        x1, y1, x2, y2 = map(int, player_bbox)
        
        # Add padding around the bbox to ensure number is captured
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
        
        player_region = frame[y1:y2, x1:x2]
        result = self.reader.ocr(player_region, cls=False)
        
        if result and len(result) > 0:
            # Process results in pairs (bbox, text_info)
            for i in range(0, len(result[0]), 2):
                # Skip bbox (result[0][i]) and get text_info
                text_info = result[0][i + 1]
                if isinstance(text_info, tuple):
                    text, confidence = text_info
                    if str(text).isdigit() and confidence > 0.5:
                        return int(text), confidence  # Return both number and confidence
        
        return None, 0.0  # Return None and 0 confidence if no number found

    def detect_numbers_in_frame(self, frame, player_tracks):
        """Detect jersey numbers for all players in a frame"""
        jersey_numbers = {}
        for player_id, track_info in player_tracks.items():
            bbox = track_info["bbox"]
            number, confidence = self.get_jersey_number(frame, bbox)  # Get both number and confidence
            if number:
                jersey_numbers[player_id] = (number, confidence)  # Store both as a tuple
        return jersey_numbers 