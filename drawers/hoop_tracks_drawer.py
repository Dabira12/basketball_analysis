import cv2
import numpy as np

class HoopTracksDrawer:
    """
    A drawer class responsible for drawing hoop locations on video frames.

    Attributes:
        hoop_color (tuple): The color used to draw the hoop markers (in BGR format).
        corner_length (int): Length of corner markers in pixels.
        thickness (int): Thickness of the drawing lines.
    """

    def __init__(self):
        """
        Initialize the HoopTracksDrawer instance with default settings.
        """
        self.hoop_color = (0, 0, 255)  # Red color for the hoop
        self.corner_length = 20  # Length of corner markers
        self.thickness = 2

    def draw_hoop_corners(self, frame, bbox):
        """
        Draws corner markers to indicate a hoop.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw corners (like a basketball hoop backboard)
        # Top left
        cv2.line(frame, (x1, y1), (x1 + self.corner_length, y1), self.hoop_color, self.thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + self.corner_length), self.hoop_color, self.thickness)
        
        # Top right
        cv2.line(frame, (x2, y1), (x2 - self.corner_length, y1), self.hoop_color, self.thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + self.corner_length), self.hoop_color, self.thickness)
        
        # Bottom left
        cv2.line(frame, (x1, y2), (x1 + self.corner_length, y2), self.hoop_color, self.thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - self.corner_length), self.hoop_color, self.thickness)
        
        # Bottom right
        cv2.line(frame, (x2, y2), (x2 - self.corner_length, y2), self.hoop_color, self.thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - self.corner_length), self.hoop_color, self.thickness)
        
        # Draw rim line (horizontal line at 2/3 height)
        rim_y = int(y1 + (y2 - y1) * 2/3)
        cv2.line(frame, (x1, rim_y), (x2, rim_y), self.hoop_color, self.thickness)

        return frame

    def draw(self, video_frames, tracks):
        """
        Draws hoop markers on each video frame based on provided detection information.

        Args:
            video_frames (list): A list of video frames (as NumPy arrays or image objects).
            tracks (list): A list of dictionaries containing hoop tracking information.

        Returns:
            list: A list of processed video frames with drawn hoop markers.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Get hoop data - matches format from hoop_tracker.py
            hoop_dict = tracks[frame_num]
            if 1 in hoop_dict and "bbox" in hoop_dict[1]:  # Check for hoop ID 1
                bbox = hoop_dict[1]["bbox"]
                frame = self.draw_hoop_corners(frame, bbox)

            output_video_frames.append(frame)
            
        return output_video_frames
