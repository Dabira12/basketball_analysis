from ultralytics import YOLO
import supervision as sv
import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class HoopTracker:
    """
    A class that handles basketball hoop detection using YOLO.
    Since hoops are typically stationary, we can simplify the tracking aspect.
    """
    def __init__(self, model_path):
        """
        Initialize the HoopTracker with YOLO model.

        Args:
            model_path (str): Path to the YOLO model weights trained for hoop detection.
        """
        self.model = YOLO(model_path)
        
    def detect_frames(self, frames):
        """
        Detect hoops in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get hoop tracking results for a sequence of frames with optional caching.
        This method name matches the interface used by other trackers.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing hoop locations for each frame,
                where each dictionary contains the hoop's bounding box coordinates.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            tracks.append({})
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()  # Access bbox using index like in ball_tracker
                cls_id = frame_detection[3]  # Access class_id using index
                
                if cls_id == cls_names_inv['Hoop']:  # Assuming 'Hoop' is the class name
                    tracks[frame_num][1] = {"bbox": bbox}  # Using ID 1 like in ball_tracker
        
        save_stub(stub_path, tracks)
        return tracks
