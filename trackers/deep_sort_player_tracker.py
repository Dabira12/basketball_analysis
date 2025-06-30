from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from ultralytics import YOLO
import numpy as np
import torch
from pathlib import Path
import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class DeepSORTPlayerTracker:
    """
    A class that handles player detection and tracking using YOLO and DeepOCSORT.

    This class combines YOLO object detection with DeepOCSORT tracking to maintain consistent
    player identities across frames while processing detections in batches.
    """
    def __init__(self, model_path, reid_model_path="osnet_x0_25_msmt17.pt", stub_path=None):
        """
        Initialize the DeepSORTPlayerTracker with YOLO model and DeepOCSORT tracker.
        
        Args:
            model_path (str): Path to the YOLO model weights.
            reid_model_path (str): Path to the ReID model weights for appearance matching.
            stub_path (str, optional): Path to save/load detection stubs.
        """
        self.model = YOLO(model_path)
        self.tracker = DeepOcSort(
            reid_weights=Path(reid_model_path),  # will be downloaded automatically
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            half=True
        )
        self.stub_path = stub_path

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = 20
        detections = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            results = self.model(batch, verbose=False)
            
            for j, result in enumerate(results):
                frame_idx = i + j
                dets = result.boxes.data.cpu().numpy()  # Get detections as numpy array
                if len(dets) > 0:
                    tracked = self.tracker.update(dets, frames[frame_idx])
                    detections.append(tracked)
                else:
                    detections.append(np.array([]))
                    
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get object tracks from a batch of frames.
        
        This method processes frames in batches to detect and track players. It first attempts
        to read pre-computed tracks from a stub file if read_from_stub is True.
        
        Args:
            frames (list): List of frames to process
            read_from_stub (bool, optional): Whether to try reading from stub. Defaults to False.
            stub_path (str, optional): Path to the stub file. Defaults to None.
            
        Returns:
            list: List of dictionaries containing track information for each frame.
                 Each dictionary maps player IDs to their track information.
        """
        # Try to read from stub first if enabled
        if read_from_stub and stub_path:
            stub_tracks = read_stub(stub_path)
            if stub_tracks is not None:
                return stub_tracks

        tracks_list = []
        batch_size = 20
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            results = self.model(batch_frames, verbose=False)
            
            for frame, result in zip(batch_frames, results):
                # Convert YOLO results to the format expected by DeepOCSORT
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()
                
                # Get class names mapping
                cls_names = result.names
                cls_names_inv = {v:k for k,v in cls_names.items()}
                
                # Combine into detection array (x1, y1, x2, y2, conf, cls)
                detections = np.column_stack((boxes, confs, clss))
                
                if len(detections) > 0:
                    tracks = self.tracker.update(detections, frame)
                else:
                    tracks = np.empty((0, 8))
                
                # Convert tracks to dictionary format
                tracks_dict = {}
                for track in tracks:
                    player_id = int(track[4])  # track_id
                    cls_id = int(track[6])     # class id
                    
                    # Simple class check like in PlayerTracker
                    if cls_id == cls_names_inv['Player']:
                        tracks_dict[player_id] = {
                            'bbox': track[0:4],  # x1, y1, x2, y2
                            'conf': track[5],    # confidence
                            'class': track[6]    # class id
                        }
                
                tracks_list.append(tracks_dict)
        
        # Save to stub if path provided
        if stub_path:
            save_stub(stub_path, tracks_list)
            
        return tracks_list 