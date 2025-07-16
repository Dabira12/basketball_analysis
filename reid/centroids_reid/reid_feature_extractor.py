# basketball_analysis/reid_feature_extractor.py
from pathlib import Path
import sys
import os
import numpy as np
import torch
import cv2
from PIL import Image
from reid.centroids_reid.config import cfg
from reid.centroids_reid.train_ctl_model import CTLModel
from reid.centroids_reid.datasets.transforms import ReidTransforms

class ReIDFeatureExtractor:
    def __init__(self, model_version='res50_market'):
        # Get absolute path to centroids_reid directory
        CENTROIDS_ROOT = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.dirname(CENTROIDS_ROOT))  # Add reid to path
        
        # Model configs with correct paths
        self.ver_to_specs = {
            "res50_market": (
                os.path.join(CENTROIDS_ROOT, "configs/256_resnet50.yml"),
                os.path.join(CENTROIDS_ROOT, "models/market1501_resnet50_256_128_epoch_120.ckpt")
            ),
            "res50_duke": (
                os.path.join(CENTROIDS_ROOT, "configs/256_resnet50.yml"), 
                os.path.join(CENTROIDS_ROOT, "models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt")
            )
        }
        
        # Load model
        CONFIG_FILE, MODEL_FILE = self.ver_to_specs[model_version]
        cfg.merge_from_file(CONFIG_FILE)
        opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, 
                "MODEL.PRETRAINED", True, 
                "TEST.ONLY_TEST", True, 
                "MODEL.RESUME_TRAINING", False]
        cfg.merge_from_list(opts)
        
        self.use_cuda = torch.cuda.is_available() and cfg.GPU_IDS
        self.model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)
        if self.use_cuda:
            self.model.to('cuda')
        self.model.eval()
        
        # Setup transforms
        transforms_base = ReidTransforms(cfg)
        self.transforms = transforms_base.build_transforms(is_train=False)
        
    def process_tracklet(self, frames, player_tracks, output_folder):
        """Extract and save features for all players"""
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Group frames by player_id (player_tracks is already in this format)
        player_tracklets = {}
        
        # Process each player's tracks
        for player_id, frame_tracks in player_tracks.items():
            player_tracklets[player_id] = []
            
            # Process each frame for this player
            for frame_idx, track_info in frame_tracks.items():
                # Save crop to disk
                x1, y1, x2, y2 = map(int, track_info["bbox"])
                crop = frames[frame_idx][y1:y2, x1:x2]
                
                track_dir = Path(output_folder) / str(player_id)
                track_dir.mkdir(exist_ok=True)
                img_path = track_dir / f"frame_{frame_idx}.jpg"
                cv2.imwrite(str(img_path), crop)
                player_tracklets[player_id].append(str(img_path))
        
        # Extract features for each player
        for player_id, image_paths in player_tracklets.items():
            features = []
            for img_path in image_paths:
                img = cv2.imread(img_path)
                input_img = Image.fromarray(img)
                input_img = torch.stack([self.transforms(input_img)])
                
                with torch.no_grad():
                    _, global_feat = self.model.backbone(input_img.cuda() if self.use_cuda else input_img)
                    global_feat = self.model.bn(global_feat)
                features.append(global_feat.cpu().numpy().reshape(-1,))
            
            # Save features
            np_feat = np.array(features)
            feature_path = Path(output_folder) / f"{player_id}_features.npy"
            with open(feature_path, 'wb') as f:
                np.save(f, np_feat)
        
        return player_tracklets