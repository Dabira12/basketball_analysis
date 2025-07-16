import os
import argparse
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, DeepSORTPlayerTracker, HoopTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from jersey_number_detector_new import JerseyNumberDetectorNew
from drawers import (
    PlayerTracksDrawer, 
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer,
    HoopTracksDrawer
)
from configs import(
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)
import logging
import json

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Read Video
    video_frames = read_video(args.input_video)
    
    ## Initialize Trackers and Detectors
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    # player_tracker = DeepSORTPlayerTracker(PLAYER_DETECTOR_PATH, reid_model_path="osnet_x0_25_msmt17.pt"  )
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)
    # jersey_detector = JerseyNumberDetector()
    jersey_detector = JerseyNumberDetectorNew()
    hoop_tracker = HoopTracker(BALL_DETECTOR_PATH)
    ## Initialize Keypoint Detector
    # court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    # Run Detectors
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path=os.path.join(args.stub_path, 'player_track_stubs.pkl')
                                      )
    hoop_tracks = hoop_tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path=os.path.join(args.stub_path, 'hoop_track_stubs.pkl')
                                      )
    
    # Detect jersey numbers for each player

    print("Detecting jersey numbers")
    jersey_numbers = {}
    jersey_numbers = jersey_detector.process_tracklet(video_frames, player_tracks)
    
    # Add logging
    logger.info("=" * 50)
    logger.info("Jersey Number Detection Results:")
    logger.info("=" * 50)
    
    # Log formatted results
    if not jersey_numbers:
        logger.warning("No jersey numbers detected!")
    else:
        for player_id, (number, confidence) in jersey_numbers.items():
            logger.info(f"Player {player_id:3d}: Jersey #{number:3d} (confidence: {confidence:.2f})")
    
    # Log full data structure
    logger.info("\nFull Data Structure:")
    logger.info("-" * 50)
    import json
    logger.info(f"Type: {type(jersey_numbers)}")
    logger.info("Contents:")
    try:
        # Pretty print the dictionary with indentation
        formatted_data = json.dumps(jersey_numbers, indent=2)
        logger.info(f"\n{formatted_data}")
    except TypeError:
        # If json.dumps fails (e.g., due to numpy arrays or other non-serializable types)
        logger.info("Raw structure:")
        for player_id, data in jersey_numbers.items():
            logger.info(f"  {player_id}: {data}")
    
    logger.info("=" * 50)

    # for frame_idx, frame in enumerate(video_frames):
    #     print(f"Frame {frame_idx}")
    #     frame_numbers = jersey_detector.detect_numbers_in_frame(frame, player_tracks[frame_idx], frame_idx)
    #     for player_id, (number, confidence) in frame_numbers.items():
    #         if player_id not in jersey_numbers:
    #             jersey_numbers[player_id] = (number, confidence)
    #         elif confidence > jersey_numbers[player_id][1]:  # Only check confidence if player_id exists
    #             jersey_numbers[player_id] = (number, confidence)



    print(jersey_numbers)
    
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                 read_from_stub=False,
                                                 stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl')
                                                )
    ## Run KeyPoint Extractor
    # court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(video_frames,
    #                                                                 read_from_stub=True,
    #                                                                 stub_path=os.path.join(args.stub_path, 'court_key_points_stub.pkl')
    #                                                                 )

    # Remove Wrong Ball Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    # Interpolate Ball Tracks
    # ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
   

    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                    player_tracks,
                                                                    read_from_stub=False,
                                                                    stub_path=os.path.join(args.stub_path, 'player_assignment_stub.pkl')
                                                                    )

    # Ball Acquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)

    # # Detect Passes
    # pass_and_interception_detector = PassAndInterceptionDetector()
    # passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    # interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)

    # # Tactical View
    # tactical_view_converter = TacticalViewConverter(
    #     court_image_path="./images/basketball_court.png"
    # )

    # court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    # tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints_per_frame,player_tracks)

    # Speed and Distance Calculator
    # speed_and_distance_calculator = SpeedAndDistanceCalculator(
    #     tactical_view_converter.width,
    #     tactical_view_converter.height,
    #     tactical_view_converter.actual_width_in_meters,
    #     tactical_view_converter.actual_height_in_meters
    # )
    # player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    # player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)

    # Draw output   
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    hoop_tracks_drawer = HoopTracksDrawer()
    # court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    # pass_and_interceptions_drawer = PassInterceptionDrawer()
    # tactical_view_drawer = TacticalViewDrawer()
    # speed_and_distance_drawer = SpeedAndDistanceDrawer()

    ## Draw object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, 
                                                    player_tracks,
                                                    player_assignment,
                                                    ball_aquisition,
                                                    jersey_numbers)  # Add jersey numbers to visualization
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    ## Draw KeyPoints
    # output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints_per_frame)

    ## Draw Frame Number
    output_video_frames = frame_number_drawer.draw(output_video_frames)

    # Draw Team Ball Control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                        player_assignment,
                                                        ball_aquisition)

    # Draw Hoops
    print(hoop_tracks)
    output_video_frames = hoop_tracks_drawer.draw(output_video_frames, hoop_tracks)

    # # Draw Passes and Interceptions
    # output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
    #                                                          passes,
    #                                                          interceptions)
    
    # Speed and Distance Drawer
    # output_video_frames = speed_and_distance_drawer.draw(output_video_frames,
    #                                                      player_tracks,
    #                                                      player_distances_per_frame,
    #                                                      player_speed_per_frame
    #                                                      )

    # ## Draw Tactical View
    # output_video_frames = tactical_view_drawer.draw(output_video_frames,
    #                                                 tactical_view_converter.court_image_path,
    #                                                 tactical_view_converter.width,
    #                                                 tactical_view_converter.height,
    #                                                 tactical_view_converter.key_points,
    #                                                 tactical_player_positions,
    #                                                 player_assignment,
    #                                                 ball_aquisition,
    #                                                 )

    # Save video
    save_video(output_video_frames, args.output_video)

if __name__ == '__main__':
    main()
    