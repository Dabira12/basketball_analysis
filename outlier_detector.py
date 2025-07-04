class OutlierDetector:
    def __init__(self, threshold=3.5, rounds=3):
        self.threshold = threshold
        self.rounds = rounds
    
    def remove_outliers(self, image_folder, feature_folder):
        """Matches original gaussian_outliers.py functionality"""
        tracks = []
        for track in os.listdir(image_folder):
            if os.path.exists(os.path.join(feature_folder, f"{track}_features.npy")):
                tracks.append(track)
                
        results = {}
        for r in range(self.rounds):
            results[r] = {x: [] for x in tracks}
            
        for tr in tracks:
            images = os.listdir(os.path.join(image_folder, tr))
            features_path = os.path.join(feature_folder, f"{tr}_features.npy")
            
            with open(features_path, 'rb') as f:
                features = np.load(f)
                
            if len(images) <= 2:
                results[tr] = images
                continue
                
            cleaned_data = features
            for r in range(self.rounds):
                mu = np.mean(cleaned_data, axis=0)
                euclidean_distance = np.linalg.norm(features - mu, axis=1)
                mean_euclidean_distance = np.mean(euclidean_distance)
                std = np.std(euclidean_distance)
                
                cleaned_data = features[(euclidean_distance - mean_euclidean_distance) <= self.threshold]
                cleaned_data_indexes = np.where((euclidean_distance - mean_euclidean_distance) <= self.threshold)[0]
                
                for i in cleaned_data_indexes:
                    results[r][tr].append(images[i])
                    
        # Save results
        for r in range(self.rounds):
            result_file_name = f"main_subject_gauss_th={self.threshold}_r={r + 1}.json"
            with open(os.path.join(feature_folder, result_file_name), "w") as outfile:
                json.dump(results[r], outfile)
                
        return results