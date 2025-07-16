# basketball_analysis/jersey_number_detector_new.py

from reid.centroids_reid.reid_feature_extractor import ReIDFeatureExtractor
from outlier_detector import OutlierDetector
import cv2
import numpy as np
from str.parseq.strhub.models.utils import create_model
import torch
import logging
import os
from datetime import datetime
import traceback  # Add this import here
from pathlib import Path # Added for Path

# Add at the top of the file, after existing imports
import logging
import os
from datetime import datetime


# Add right after class definition, before __init__
def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"jersey_detection_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class JerseyNumberDetectorNew:
    def __init__(self):
        # Initialize only the basic components
        self._reid_extractor = None
        self.outlier_detector = OutlierDetector()
        # Remove PaddleOCR initialization
        self._parseq_model = None  # Lazy load the STR model
        self._pose_model = None
        
        # Set up logging
        self.logger = setup_logging()
        self.logger.info("Initializing Jersey Number Detector")

    @property
    def reid_extractor(self):
        if self._reid_extractor is None:
            try:
                from reid.centroids_reid.reid_feature_extractor import ReIDFeatureExtractor
                self._reid_extractor = ReIDFeatureExtractor()
            except ImportError as e:
                print(f"ReID functionality not available: {e}")
                self._reid_extractor = None
        return self._reid_extractor

    @property
    def parseq_model(self):
        """Lazy loading of the PARSeq model"""
        if self._parseq_model is None:
            try:
                # Add STR module to Python path
                import sys
                str_path = "/workspace/basketball_analysis/str"
                if str_path not in sys.path:
                    sys.path.append(str_path)
                    self.logger.info(f"Added {str_path} to Python path")
                
                self.logger.info("Attempting to import create_model")
                # Import and initialize PARSeq model
                from str.parseq.strhub.models.utils import create_model
                self.logger.info("Successfully imported create_model")
                
                import torch
                self.logger.info("Checking model path...")
                # model_path = '/workspace/basketball_analysis/models/arseq_hockey.pth.ckpt'
                model_path = '/workspace/basketball_analysis/models/soccernet.pth.ckpt'
                if not os.path.exists(model_path):
                    self.logger.error(f"Model file not found at: {model_path}")
                    return None
                
                self.logger.info(f"Loading model from: {model_path}")
                # Create model with exact config to match checkpoint
                self._parseq_model = create_model(
                    'parseq',
                    pretrained=model_path,
                    max_label_length=25,  # Set to 25 since model adds +1 internally for EOS token
                    decode_ar=True,
                    refine_iters=1
                )
                self.logger.info("Model created successfully")
                
                self._parseq_model.eval()  # Set to evaluation mode
                if torch.cuda.is_available():
                    self._parseq_model = self._parseq_model.cuda()
                    self.logger.info("Model moved to CUDA")
                else:
                    self.logger.info("CUDA not available, using CPU")
                    
            except ImportError as e:
                self.logger.error(f"Failed to import required modules: {e}")
                self._parseq_model = None
            except Exception as e:
                self.logger.error(f"Failed to initialize STR model: {e}")
                self.logger.error(f"Error type: {type(e)}")
                self._parseq_model = None
        return self._parseq_model

    def init_pose_model(self):
        """Initialize ViTPose model for keypoint detection"""
        try:
            from mmpose.apis import init_pose_model
            
            # config_file = "pose/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py"
            config_file = "pose/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py"
            checkpoint = "pose/ViTPose/checkpoints/vitpose-h.pth"
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            pose_model = init_pose_model(config_file, checkpoint, device=device)
            
            self.logger.info("Successfully initialized pose model")
            return pose_model
        except Exception as e:
            self.logger.error(f"Failed to initialize pose model: {str(e)}")
            return None

    def get_jersey_roi_from_pose(self, frame, bbox, keypoints):
        """Extract jersey ROI using pose keypoints"""
        self.logger.info("Attempting pose-guided ROI extraction")
        
        # Get relevant keypoints (shoulders and hips)
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        self.logger.info(f"Keypoints found:"
                        f"\n  Left shoulder: {left_shoulder}"
                        f"\n  Right shoulder: {right_shoulder}"
                        f"\n  Left hip: {left_hip}"
                        f"\n  Right hip: {right_hip}")

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            self.logger.warning("Missing keypoints, falling back to basic crop")
            return self.get_basic_crop(frame, bbox, track_id, frame_idx)

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

        # self.logger.info(f"Jersey ROI coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        # self.logger.info(f"Crop shape: {crop.shape}")
        return crop

    def get_basic_crop(self, frame, bbox, track_id, frame_idx):
        """Fallback method for basic bbox cropping focusing on center-upper torso area"""
        self.logger.info(f"Using basic crop with bbox: {bbox}")
        
        # Create output directory structure
        output_dir = Path("crops") / str(track_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get original bbox coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate dimensions
        height = y2 - y1
        width = x2 - x1
        
        # Adjust to focus on center-upper torso
        torso_top = y1 + int(height * 0.15)      # Start 15% down from top
        torso_bottom = y1 + int(height * 0.55)    # End 55% down
        
        # Trim width from both sides (reduce by 15% from each side)
        width_reduction = int(width * 0.15)       # 15% reduction from each side
        x1 = x1 + width_reduction                 # Move right from left edge
        x2 = x2 - width_reduction                 # Move left from right edge
        
        # Add small padding
        padding = 5
        x1 = max(0, x1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y1 = max(0, torso_top - padding)
        y2 = min(frame.shape[0], torso_bottom + padding)
        
        # self.logger.info(f"Torso crop coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        crop = frame[y1:y2, x1:x2]
        # self.logger.info(f"Crop shape: {crop.shape}")
        
        # Save crop
        output_path = output_dir / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(output_path), crop)
        self.logger.info(f"Saved torso crop to: {output_path}")
        
        return crop

    def get_number_prediction(self, crop):
        """Get jersey number using STR model"""
        if self.parseq_model is None:
            self.logger.warning("STR model not initialized")
            return None, 0.0
        
        try:
            import torch
            import torchvision.transforms as T
            
            # Preprocess image
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((32, 128)),  # Standard size for STR models
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
            
            # Convert crop to tensor and preprocess
            img_tensor = transform(crop).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # Get prediction
            with torch.no_grad():
                logits = self.parseq_model(img_tensor)  # Model returns logits directly
                probs = logits.softmax(-1)  # Convert to probabilities
                preds, confidences = self.parseq_model.tokenizer.decode(probs)  # Returns (List[str], List[Tensor])
                
                # Enhanced logging for predictions and confidences
                self.logger.info("Raw predictions and confidences:")
                self.logger.info(f"Raw preds: {preds}")
                for i, conf in enumerate(confidences):
                    self.logger.info(f"Confidence tensor {i}: {conf}")
                    self.logger.info(f"Confidence values: {conf.cpu().numpy()}")
                
                # preds is a list of strings, take first one since batch size is 1
                pred = preds[0].strip()
                
                # Check if prediction is a number
                if pred.isdigit():
                    # Get confidence values
                    conf_values = confidences[0].cpu().numpy()
                    
                    # The model outputs confidence values for each position plus one extra value
                    # Example for single digit "1":
                    #   - First value (conf_values[0]): confidence for the digit "1"
                    #   - Second value (conf_values[1]): confidence for the End-of-Sequence (EOS) token
                    # Example for double digit "23":
                    #   - First value: confidence for "2"
                    #   - Second value: confidence for "3"
                    #   - Third value: confidence for EOS token
                    # We ignore the EOS token confidence when calculating our final confidence
                    
                    if len(pred) == 1:
                        # For single digit, take only the first confidence value
                        # Ignore the second value which is for the EOS token
                        conf = conf_values[0]
                        self.logger.info(f"Single digit {pred} - using first confidence: {conf:.3f}")
                        self.logger.info(f"(Ignored EOS token confidence: {conf_values[1]:.3f})")
                    else:
                        # For multiple digits, take average of confidence values for the digits only
                        # Ignore the last value which is for the EOS token
                        conf = conf_values[:len(pred)].mean()
                        self.logger.info(f"Multiple digits {pred} - using average of first {len(pred)} confidences: {conf:.3f}")
                        self.logger.info(f"(Ignored EOS token confidence: {conf_values[len(pred)]:.3f})")
                    
                    self.logger.info(f"Final Prediction: {pred}, Confidence: {conf:.3f}")
                    return int(pred), conf
                self.logger.info(f"Prediction: {pred} is not a number")
                return None, 0.0
                
        except Exception as e:
            self.logger.error(f"STR prediction failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, 0.0

    def add_pose_keypoints(self, frames, player_tracks):
        """Add pose keypoints to player tracks"""
        self.logger.info("Adding pose keypoints to player tracks")
        
        if not hasattr(self, 'pose_model') or self.pose_model is None:
            try:
                from mmpose.apis import init_pose_model
                config_file = "pose/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py"
                checkpoint = "pose/ViTPose/checkpoints/vitpose-h.pth"
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.pose_model = init_pose_model(config_file, checkpoint, device=device)
                self.logger.info("Successfully initialized pose model")
            except Exception as e:
                self.logger.error(f"Failed to initialize pose model: {e}")
                return player_tracks

        # Create a copy of player_tracks to avoid modifying the original
        tracks_with_pose = player_tracks.copy()
        
        # Process each frame
        for frame_idx, frame_tracks in enumerate(tracks_with_pose):
            self.logger.info(f"Processing frame {frame_idx}")
            
            # Process each player in the frame
            for player_id, track_info in frame_tracks.items():
                try:
                    bbox = track_info["bbox"]
                    # Convert bbox to person result format (xywh)
                    person = {'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]}
                    
                    # Run pose detection
                    from mmpose.apis import inference_top_down_pose_model
                    pose_results, _ = inference_top_down_pose_model(
                        self.pose_model,
                        frames[frame_idx],
                        [person],
                        bbox_thr=None,
                        format='xywh'
                    )
                    
                    if pose_results and len(pose_results) > 0:
                        # Add keypoints to track info
                        tracks_with_pose[frame_idx][player_id]["keypoints"] = pose_results[0]['keypoints'].tolist()
                        self.logger.info(f"Added keypoints for player {player_id}")
                    else:
                        self.logger.warning(f"No pose detected for player {player_id}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to get keypoints for player {player_id}: {e}")
                    continue
        
        return tracks_with_pose

    def process_tracklet(self, frames, player_tracks, frame_idx=None):
        """Process tracklets with ReID-based outlier removal and pose-guided ROI"""
        self.logger.info(f"Processing tracklet with {len(frames)} frames")
        
        # Convert list-based tracks to dictionary format if needed
        if isinstance(player_tracks, list):
            tracks_dict = {}
            for idx, frame_tracks in enumerate(player_tracks):
                for player_id, track in frame_tracks.items():
                    if player_id not in tracks_dict:
                        tracks_dict[player_id] = {}
                    tracks_dict[player_id][idx] = track
            player_tracks = tracks_dict
            self.logger.info(f"Converted to dictionary format. Players: {list(player_tracks.keys())}")

        # Add pose keypoints to tracks
        player_tracks = self.add_pose_keypoints(frames, player_tracks)
        self.logger.info("Added pose keypoints to tracks")

        # Create temporary directory for ReID features
        import tempfile
        import shutil
        temp_dir = None

        if self.reid_extractor is not None:
            try:
                self.logger.info("Attempting ReID processing")
                # Create temporary directory for ReID features
                temp_dir = tempfile.mkdtemp(prefix="reid_features_")
                self.logger.info(f"Created temporary directory: {temp_dir}")
                
                # Get image paths from ReID feature extractor
                self.logger.info("Starting ReID feature extraction")
                player_tracklets = self.reid_extractor.process_tracklet(frames, player_tracks, temp_dir)
                self.logger.info(f"ReID features extracted. Players: {list(player_tracklets.keys())}")
                
                # Load the saved features
                player_features = {}
                for player_id in player_tracklets:
                    try:
                        feature_path = Path(temp_dir) / f"{player_id}_features.npy"
                        if feature_path.exists():
                            features = np.load(feature_path)
                            player_features[player_id] = features
                            self.logger.info(f"Loaded features for player {player_id}, shape: {features.shape}")
                        else:
                            self.logger.warning(f"No feature file found for player {player_id}")
                    except Exception as e:
                        self.logger.error(f"Error loading features for player {player_id}: {str(e)}")
                        continue
                
                # Now pass the actual feature vectors to _process_with_reid
                return self._process_with_reid(frames, player_tracks, player_features)
            except Exception as e:
                self.logger.error(f"ReID processing failed with error: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return self._process_without_reid(frames, player_tracks)
            finally:
                if temp_dir and os.path.exists(temp_dir):
                    self.logger.info(f"Cleaning up temporary directory: {temp_dir}")
                    shutil.rmtree(temp_dir)
        else:
            self.logger.info("No ReID extractor available, using basic processing")
            return self._process_without_reid(frames, player_tracks)

    def _process_with_reid(self, frames, player_tracks, player_features):
        """Process using ReID features"""
        final_results = {}
        self.logger.info(f"Starting ReID processing with {len(player_features)} players")
        
        # First validate that all players in features exist in tracks
        missing_players = []
        for player_id in player_features.keys():
            if player_id not in player_tracks:
                self.logger.error(f"Player {player_id} found in features but missing from tracks")
                missing_players.append(player_id)
        
        # Log track information
        self.logger.info(f"Available players in tracks: {list(player_tracks.keys())}")
        self.logger.info(f"Available players in features: {list(player_features.keys())}")
        
        for player_id, features in player_features.items():
            # Skip players that don't have tracking data
            if player_id in missing_players:
                self.logger.warning(f"Skipping player {player_id} due to missing track data")
                continue
            
            self.logger.info(f"\nProcessing Player {player_id} with ReID")
            
            try:
                # Log feature information
                self.logger.info(f"Feature type: {type(features)}")
                if isinstance(features, np.ndarray):
                    self.logger.info(f"Feature shape: {features.shape}")
                    self.logger.info(f"Feature stats - Min: {features.min():.3f}, Max: {features.max():.3f}, Mean: {features.mean():.3f}")
                
                # Features are already numpy arrays
                self.logger.info(f"Found {len(features)} feature vectors")
                
                # Log outlier detection input
                self.logger.info(f"Running outlier detection on features")
                inlier_indices = self.outlier_detector.remove_outliers_from_features(features)
                self.logger.info(f"After outlier removal: {len(inlier_indices)} frames remaining")
                
                # Get corresponding frames/detections
                filtered_detections = []
                self.logger.info(f"Processing frames for player {player_id}")
                
                # Validate player track data
                if not player_tracks[player_id]:
                    self.logger.error(f"Empty track data for player {player_id}")
                    continue
                    
                track_frames = list(player_tracks[player_id].keys())
                self.logger.info(f"Available frame indices for player {player_id}: {track_frames}")
                
                # Validate indices before processing
                valid_indices = []
                for idx in inlier_indices:
                    if idx < len(track_frames):
                        valid_indices.append(idx)
                    else:
                        self.logger.warning(f"Index {idx} out of range for player {player_id}'s track frames")
                
                for idx in valid_indices:
                    try:
                        frame_idx = track_frames[idx]
                        track_info = player_tracks[player_id][frame_idx]
                        
                        self.logger.info(f"Processing frame {frame_idx}")
                        # self.logger.info(f"Track info: {track_info}")
                        
                        # Get pose keypoints if available
                        if hasattr(track_info, 'keypoints'):
                            # Use pose-guided ROI if keypoints are available
                            crop = self.get_jersey_roi_from_pose(frames[frame_idx], 
                                                               track_info["bbox"],
                                                               track_info["keypoints"])
                        else:
                            # Fall back to basic crop if no keypoints
                            self.logger.warning("No keypoints available, using basic crop")
                            crop = self.get_basic_crop(frames[frame_idx], track_info["bbox"], player_id, frame_idx)
                            
                        self.logger.info(f"Crop shape: {crop.shape}")
                        
                        filtered_detections.append({
                            'frame_idx': frame_idx,
                            'crop': crop
                        })
                    except Exception as e:
                        self.logger.error(f"Error processing frame {idx} for player {player_id}: {str(e)}")
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                
                # Process filtered detections
                predictions = []
                for det in filtered_detections:
                    number, conf = self.get_number_prediction(det['crop'])
                    if number is not None:
                        self.logger.info(f"Frame {det['frame_idx']}: Detected number {number} (conf: {conf:.3f})")
                        predictions.append((number, conf))
                    else:
                        self.logger.info(f"Frame {det['frame_idx']}: No number detected")
                
                # Consolidate predictions
                if predictions:
                    final_number, confidence = self.consolidate_predictions(predictions)
                    if final_number is not None:
                        self.logger.info(f"Final number for Player {player_id}: {final_number} (conf: {confidence:.3f})")
                        final_results[player_id] = (final_number, confidence)
                    else:
                        self.logger.info(f"Could not determine final number for Player {player_id}")
                else:
                    self.logger.info(f"No valid predictions for Player {player_id}")
                    
            except Exception as e:
                self.logger.error(f"Error processing player {player_id}: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        return final_results

    def _process_without_reid(self, frames, player_tracks):
        """Fallback processing without ReID"""
        self.logger.info("Processing without ReID")
        final_results = {}
        
        for player_id, frame_tracks in player_tracks.items():
            self.logger.info(f"\nProcessing Player {player_id}")
            predictions = []
            
            for frame_idx, track in frame_tracks.items():
                self.logger.info(f"\nFrame {frame_idx}:")
                self.logger.info(f"  Track info: {track}")
                
                # Log frame shape
                self.logger.info(f"  Frame shape: {frames[frame_idx].shape}")
                
                # Get crop
                crop = self.get_basic_crop(frames[frame_idx], track["bbox"], player_id, frame_idx)
                
                # Get prediction
                number, conf = self.get_number_prediction(crop)
                if number is not None:
                    self.logger.info(f"  Detected number {number} with confidence {conf:.3f}")
                    predictions.append((number, conf))
                else:
                    self.logger.info("  No number detected")
            
            # Consolidate predictions for this player
            if predictions:
                self.logger.info(f"\nConsolidating {len(predictions)} predictions for Player {player_id}:")
                for num, conf in predictions:
                    self.logger.info(f"  Number: {num}, Confidence: {conf:.3f}")
                
                final_number, confidence = self.consolidate_predictions(predictions)
                if final_number is not None:
                    self.logger.info(f"Final number for Player {player_id}: {final_number} (conf: {confidence:.3f})")
                    final_results[player_id] = (final_number, confidence)
                else:
                    self.logger.info(f"Could not determine final number for Player {player_id}")
            else:
                self.logger.info(f"No valid predictions for Player {player_id}")
        
        self.logger.info(f"\nProcessing complete. Found numbers for {len(final_results)} players")
        return final_results

    # def consolidate_predictions(self, predictions):
    #     """Consolidate multiple predictions for a tracklet"""
    #     if not predictions:
    #         self.logger.debug("No predictions to consolidate")
    #         return None, 0.0
        
    #     self.logger.info(f"Consolidating {len(predictions)} predictions")
    #     number_stats = {}
    #     for number, conf in predictions:
    #         if number not in number_stats:
    #             number_stats[number] = {'count': 1, 'total_conf': conf, 'max_conf': conf}
    #         else:
    #             number_stats[number]['count'] += 1
    #             number_stats[number]['total_conf'] += conf
    #             number_stats[number]['max_conf'] = max(number_stats[number]['max_conf'], conf)
        
    #     # Log statistics for each number
    #     for number, stats in number_stats.items():
    #         self.logger.info(f"Number {number}: detected {stats['count']} times, max conf: {stats['max_conf']:.3f}")
        
    #     # Choose number based on:
    #     # 1. Frequency of detection
    #     # 2. Average confidence
    #     best_number = max(
    #         number_stats.items(),
    #         key=lambda x: (
    #             x[1]['count'],  # First priority: number of detections
    #             x[1]['total_conf'] / x[1]['count']  # Second priority: average confidence
    #         )
    #     )
        
    #     number = best_number[0]
    #     avg_conf = best_number[1]['total_conf'] / best_number[1]['count']
        
    #     # Only return if we have enough detections and confidence
    #     if best_number[1]['count'] >= 3 and avg_conf > 0.5:
    #         return number, avg_conf
    #     return None, 0.0


    def consolidate_predictions(self, predictions):
        """Consolidate multiple predictions for a tracklet"""
        if not predictions:
            self.logger.debug("No predictions to consolidate")
            return None, 0.0
        
        self.logger.info(f"Consolidating {len(predictions)} predictions")
        number_stats = {}
        total_predictions = len(predictions)
        
        # Calculate detection stats with weighted confidence
        for number, conf in predictions:
            if number not in number_stats:
                number_stats[number] = {
                    'count': 1, 
                    'total_conf': conf, 
                    'max_conf': conf,
                    'weighted_conf': conf  # Start weighted confidence
                }
            else:
                number_stats[number]['count'] += 1
                number_stats[number]['total_conf'] += conf
                number_stats[number]['max_conf'] = max(number_stats[number]['max_conf'], conf)
                # Give more weight to higher confidence detections
                number_stats[number]['weighted_conf'] = (
                    0.7 * number_stats[number]['weighted_conf'] + 
                    0.3 * conf * (number_stats[number]['count'] / total_predictions)
                )
        
        # Log statistics for each number
        for number, stats in number_stats.items():
            self.logger.info(
                f"Number {number}: detected {stats['count']} times, "
                f"max conf: {stats['max_conf']:.3f}, "
                f"weighted conf: {stats['weighted_conf']:.3f}"
            )
        
        # Choose number based on:
        # 1. Frequency of detection
        # 2. Weighted confidence (combines average conf and detection frequency)
        # 3. Maximum confidence as tiebreaker
        best_number = max(
            number_stats.items(),
            key=lambda x: (
                x[1]['count'],  # First priority
                x[1]['weighted_conf'],  # Second priority
                x[1]['max_conf']  # Third priority (tiebreaker)
            )
        )
        
        number = best_number[0]
        stats = best_number[1]
        
        # Dynamic thresholds based on total predictions
        min_detections = max(3, int(total_predictions * 0.15))  # At least 15% of frames
        min_conf = 0.4 if stats['max_conf'] > 0.8 else 0.5  # Lower threshold if we have high confidence detections
        
        # Return if we meet either:
        # 1. Enough detections with decent confidence
        # 2. Fewer detections but very high confidence
        if ((stats['count'] >= min_detections and stats['weighted_conf'] > min_conf) or
            (stats['count'] >= 2 and stats['max_conf'] > 0.85)):
            return number, stats['weighted_conf']
        
        return None, 0.0