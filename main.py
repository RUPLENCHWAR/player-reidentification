import os
import cv2
import torch
from tracker import PlayerTracker
from video_utils import download_model, annotate_frame
from ultralytics import YOLO

def main():
    print("Starting player re-identification...")
    
    # Configuration
    VIDEO_PATH = "15sec_input_720p.mp4"
    MODEL_PATH = "best.pt"
    OUTPUT_PATH = "output/tracked_video.mp4"
    
    print(f"Using video: {VIDEO_PATH}")
    print(f"Using model: {MODEL_PATH}")
    
    # Setup
    os.makedirs("output", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        download_model()
    else:
        print("Model already exists")
    
    # Load model
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO(MODEL_PATH)
    model.to(device)
    print("Model loaded successfully")
    
    # Initialize tracker
    tracker = PlayerTracker(max_disappeared=30)
    
    # Video processing
    print("Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video specs: {frame_width}x{frame_height} @ {fps}fps")
    
    out = cv2.VideoWriter(OUTPUT_PATH, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, 
                         (frame_width, frame_height))
    
    frame_count = 0
    print("Starting processing...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...")
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Process detections
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                cls = box.cls.item()
                detections.append([x1, y1, x2, y2, conf, cls])
        
        # Filter for players (class 0)
        player_detections = [det for det in detections if det[5] == 0]
        print(f"Frame {frame_count}: Found {len(player_detections)} players")
        
        # Track players
        tracked_players = tracker.update(player_detections)
        
        # Visualize
        output_frame = annotate_frame(frame, tracked_players)
        out.write(output_frame)
    
    print(f"Finished processing {frame_count} frames")
    cap.release()
    out.release()
    print(f"Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    print("Script started")
    main()
    print("Script completed")