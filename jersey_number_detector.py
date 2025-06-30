import cv2
import numpy as np
from paddleocr import PaddleOCR

import logging
import os
from datetime import datetime

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a log file with timestamp
log_filename = os.path.join(log_dir, f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class JerseyNumberDetector:
    def __init__(self):
        logging.info("Downloading PaddleOCR files...")

        # Initialize PaddleOCR with specific models
        self.reader = PaddleOCR(
            lang='en',  # Use English model
            use_angle_cls=True,
            show_log=True,
            device='gpu',
            gpu_mem=1000,
            enable_mkldnn=True,
            det_db_thresh=0.3,  # Detection threshold
            det_db_box_thresh=0.6,  # Box threshold
            det_model_dir='en_PP-OCRv3_det_slim_infer',  # Use default detection model
            rec_model_dir='en_PP-OCRv3_rec_slim_infer',  # Use default recognition model
            # text_detection_model_dir='PP-OCRv5_server_det',  # Latest detection model
            # text_recognition_model_dir='PP-OCRv5_server_rec',
            # text_recognition_model_name='PP-OCRv5_server_rec',
            # text_detection_model_name='PP-OCRv5_server_det'  # Latest recognition model
        )
        
    def get_jersey_number(self, frame, player_bbox, frame_number):
        logging.info(f"Frame {frame_number} - Getting jersey number for player {player_bbox}")
        """Extract jersey number from a player's bounding box"""
        x1, y1, x2, y2 = map(int, player_bbox)
        
        # Add padding around the bbox to ensure number is captured
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
        
        player_region = frame[y1:y2, x1:x2]
        result = self.reader.ocr(player_region)

        print(f"Frame {frame_number} - OCR Result: {result}")
        logging.info(f"Frame {frame_number} - OCR Result: {result}")
    
        if not result or not result[0]:
            logging.info(f"Frame {frame_number} - No text detected")
            return None, 0.0
        
        if result and len(result) > 0:
            # Process each detection directly
            for detection in result[0]:
                # Each detection already contains bbox and text_info
                text_info = detection[1]  # Second element is (text, confidence)
                if isinstance(text_info, tuple):
                    text, confidence = text_info
                    if str(text).isdigit() and confidence > 0.2:
                        print(f"Frame {frame_number} - Detected number: {int(text)} with confidence: {confidence}")
                        logging.info(f"Frame {frame_number} - Detected number: {int(text)} with confidence: {confidence}")
                        return int(text), confidence  # Return both number and confidence
        
        return None, 0.0  # Return None and 0 confidence if no number found

    def detect_numbers_in_frame(self, frame, player_tracks, frame_number):
        """Detect jersey numbers for all players in a frame"""

        print(f"Frame {frame_number} - Detecting numbers for {len(player_tracks)} players")
        logging.info(f"Frame {frame_number} - Detecting numbers for {len(player_tracks)} players")
        
        jersey_numbers = {}
        for player_id, track_info in player_tracks.items():
            bbox = track_info["bbox"]
            number, confidence = self.get_jersey_number(frame, bbox, frame_number)  # Get both number and confidence
            if number:
                jersey_numbers[player_id] = (number, confidence)  # Store both as a tuple
        return jersey_numbers 