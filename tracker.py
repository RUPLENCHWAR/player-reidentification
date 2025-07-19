import numpy as np
from collections import defaultdict

class PlayerTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.players = {}  # {id: [x1,y1,x2,y2]}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
    
    def update(self, detections):
        tracked = []
        
        # If no detections, increment disappeared counts
        if len(detections) == 0:
            for player_id in list(self.disappeared.keys()):
                self.disappeared[player_id] += 1
                if self.disappeared[player_id] > self.max_disappeared:
                    self._remove_player(player_id)
            return tracked
        
        # For new detections, match with existing players
        centroids = np.array([self._get_centroid(det) for det in detections])
        
        if len(self.players) > 0:
            # Simple nearest neighbor matching
            existing_centroids = np.array([self._get_centroid(d) for d in self.players.values()])
            distances = np.linalg.norm(existing_centroids[:, None] - centroids, axis=2)
            
            # Assign closest matches
            for i, det in enumerate(detections):
                if distances.shape[0] > 0:
                    closest = np.argmin(distances[:, i])
                    if distances[closest, i] < 100:  # Max distance threshold
                        player_id = list(self.players.keys())[closest]
                        self.players[player_id] = det
                        self.disappeared[player_id] = 0
                        tracked.append((player_id, det))
                        distances[closest, :] = float('inf')  # Mark as matched
                    else:
                        self._register_player(det)
                        tracked.append((self.next_id-1, det))
                else:
                    self._register_player(det)
                    tracked.append((self.next_id-1, det))
        else:
            # First frame
            for det in detections:
                self._register_player(det)
                tracked.append((self.next_id-1, det))
        
        return tracked
    
    def _get_centroid(self, detection):
        return np.array([(detection[0] + detection[2])/2, 
                        (detection[1] + detection[3])/2])
    
    def _register_player(self, detection):
        self.players[self.next_id] = detection
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def _remove_player(self, player_id):
        del self.players[player_id]
        del self.disappeared[player_id]