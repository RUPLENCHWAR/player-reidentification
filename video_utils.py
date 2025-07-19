import cv2
import gdown
import os

def annotate_frame(frame, players):
    """Draw player bounding boxes and IDs on frame"""
    annotated = frame.copy()
    for player_id, det in players:
        x1, y1, x2, y2, conf, _ = map(int, det[:6])
        color = (int(55 * player_id % 255), 
                int(100 * player_id % 255), 
                int(155 * player_id % 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"P{player_id}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return annotated

def download_model():
    """Download the custom YOLOv11 model"""
    model_url = "https://drive.google.com/uc?id=1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD"
    output_path = "best.pt"
    
    if not os.path.exists(output_path):
        gdown.download(model_url, output_path, quiet=False)
    else:
        print("Model already exists")