# basketball_analysis/jersey_number_detector_new.py

from reid.reid_feature_extractor import ReIDFeatureExtractor
from outlier_detector import OutlierDetector
from paddleocr import PaddleOCR
import cv2
import numpy as np
from str.parseq.strhub.models.utils import create_model
import torch

class JerseyNumberDetectorNew:
    def __init__(self):
        # Initialize only the basic components
        self._reid_extractor = None
        self.outlier_detector = OutlierDetector()
        # Initialize PaddleOCR for text recognition
        self.ocr = PaddleOCR(
            lang='en',
            use_angle_cls=True,
            show_log=False,
            device='gpu',
            det_model_dir='en_PP-OCRv3_det_slim_infer',
            rec_model_dir='en_PP-OCRv3_rec_slim_infer'
        )
        
        # Don't initialize ReID and other heavy components yet
        self._pose_model = None
        self._parseq_model = None

    @property
    def reid_extractor(self):
        if self._reid_extractor is None:
            try:
                from reid.reid_feature_extractor import ReIDFeatureExtractor
                self._reid_extractor = ReIDFeatureExtractor()
            except ImportError as e:
                print(f"ReID functionality not available: {e}")
                self._reid_extractor = None
        return self._reid_extractor

    def init_pose_model(self):
        """Initialize ViTPose model for keypoint detection"""
        # Similar to soccer pipeline's pose.py setup
        config_file = "pose/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py"
        checkpoint = "pose/ViTPose/checkpoints/vitpose-h.pth"
        # Initialize pose model here
        return pose_model

    def get_jersey_roi_from_pose(self, frame, bbox, keypoints):
        """Extract jersey ROI using pose keypoints
        Uses shoulders and hips to define jersey region
        """
        # Get relevant keypoints (shoulders and hips)
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return self.get_basic_crop(frame, bbox)  # Fallback to basic crop

        # Define jersey region
        x1 = min(left_shoulder[0], right_shoulder[0])
        x2 = max(left_shoulder[0], right_shoulder[0])
        y1 = min(left_shoulder[1], right_shoulder[1])
        y2 = max(left_hip[1], right_hip[1])

        # Add padding
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        return frame[int(y1):int(y2), int(x1):int(x2)]

    def get_basic_crop(self, frame, bbox):
        """Fallback method for basic bbox cropping"""
        x1, y1, x2, y2 = map(int, bbox)
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        return frame[y1:y2, x1:x2]

    def get_number_prediction(self, crop):
        """Get jersey number using PaddleOCR"""
        result = self.ocr.ocr(crop)
        if not result or not result[0]:
            return None, 0.0

        # Process OCR results
        for detection in result[0]:
            text_info = detection[1]
            if isinstance(text_info, tuple):
                text, confidence = text_info
                if str(text).isdigit() and confidence > 0.2:
                    return int(text), confidence
        return None, 0.0

    def process_tracklet(self, frames, player_tracks):
        """Process tracklets with ReID-based outlier removal and pose-guided ROI"""
        # Try to use ReID features if available
        if self.reid_extractor is not None:
            try:
                # Extract ReID features
                tracklet_features = self.reid_extractor.process_tracklet(frames, player_tracks)
                # Process with ReID features
                return self._process_with_reid(frames, player_tracks, tracklet_features)
            except Exception as e:
                print(f"ReID processing failed: {e}")
                # Fall back to basic processing
                return self._process_without_reid(frames, player_tracks)
        else:
            # Process without ReID features
            return self._process_without_reid(frames, player_tracks)

    def _process_with_reid(self, frames, player_tracks, tracklet_features):
        """Process using ReID features"""
        final_results = {}
        for player_id, features in tracklet_features.items():
            # Remove outliers using ReID features
            inlier_indices = self.outlier_detector.remove_outliers(features)
            
            # Get corresponding frames/detections
            filtered_detections = []
            for idx in inlier_indices:
                frame_idx = list(player_tracks.keys())[idx]
                track_info = player_tracks[frame_idx][player_id]
                
                # Get pose keypoints
                pose_results = self._pose_model.detect(frames[frame_idx], track_info["bbox"])
                if pose_results and pose_results[0]:
                    # Use pose-guided ROI if keypoints detected
                    crop = self.get_jersey_roi_from_pose(
                        frames[frame_idx], 
                        track_info["bbox"],
                        pose_results[0]["keypoints"]
                    )
                else:
                    # Fallback to basic crop
                    crop = self.get_basic_crop(frames[frame_idx], track_info["bbox"])
                
                filtered_detections.append({
                    'frame_idx': frame_idx,
                    'crop': crop
                })
            
            # Process filtered detections
            predictions = []
            for det in filtered_detections:
                if self.check_legibility(det['crop']):
                    number, conf = self.get_number_prediction(det['crop'])
                    if number is not None:
                        predictions.append((number, conf))
            
            # Consolidate predictions
            final_number, confidence = self.consolidate_predictions(predictions)
            if final_number is not None:
                final_results[player_id] = (final_number, confidence)
                
        return final_results

    def _process_without_reid(self, frames, player_tracks):
        """Fallback processing without ReID"""
        final_results = {}
        for player_id, track_info in player_tracks.items():
            # Process without ReID features
            # Use basic detection and OCR
            predictions = []
            for frame_idx, track in track_info.items():
                crop = self.get_basic_crop(frames[frame_idx], track["bbox"])
                number, conf = self.get_number_prediction(crop)
                if number is not None:
                    predictions.append((number, conf))
            
            # Consolidate predictions
            final_number, confidence = self.consolidate_predictions(predictions)
            if final_number is not None:
                final_results[player_id] = (final_number, confidence)
                
        return final_results

    def consolidate_predictions(self, predictions):
        """Consolidate multiple predictions for a tracklet
        Args:
            predictions: List of (number, confidence) tuples
        Returns:
            tuple: (final_number, confidence)
        """
        if not predictions:
            return None, 0.0
        
        # Count occurrences and sum confidences for each number
        number_stats = {}
        for number, conf in predictions:
            if number not in number_stats:
                number_stats[number] = {
                    'count': 1,
                    'total_conf': conf,
                    'max_conf': conf
                }
            else:
                number_stats[number]['count'] += 1
                number_stats[number]['total_conf'] += conf
                number_stats[number]['max_conf'] = max(number_stats[number]['max_conf'], conf)

        if not number_stats:
            return None, 0.0

        # Choose number based on:
        # 1. Frequency of detection
        # 2. Average confidence
        best_number = max(
            number_stats.items(),
            key=lambda x: (
                x[1]['count'],  # First priority: number of detections
                x[1]['total_conf'] / x[1]['count']  # Second priority: average confidence
            )
        )
        
        number = best_number[0]
        avg_conf = best_number[1]['total_conf'] / best_number[1]['count']
        
        # Only return if we have enough detections and confidence
        if best_number[1]['count'] >= 3 and avg_conf > 0.5:
            return number, avg_conf
        return None, 0.0