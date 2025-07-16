import numpy as np
import os
import json

class OutlierDetector:
    def __init__(self, threshold=3.5, rounds=3):
        self.threshold = threshold
        self.rounds = rounds
    
    def remove_outliers_from_features(self, features):
        """Remove outliers directly from feature array
        Args:
            features: numpy array of features
        Returns:
            indices of inlier frames
        """
        if len(features) <= 2:
            return np.arange(len(features))
            
        cleaned_data = features
        for r in range(self.rounds):
            mu = np.mean(cleaned_data, axis=0)
            euclidean_distance = np.linalg.norm(features - mu, axis=1)
            mean_euclidean_distance = np.mean(euclidean_distance)
            std = np.std(euclidean_distance)
            
            cleaned_data_indexes = np.where(
                (euclidean_distance - mean_euclidean_distance) <= self.threshold * std
            )[0]
            
            if len(cleaned_data_indexes) == 0:
                # If all data points are considered outliers, keep them all
                return np.arange(len(features))
                
            cleaned_data = features[cleaned_data_indexes]
            
        return cleaned_data_indexes