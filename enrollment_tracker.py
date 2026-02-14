
import argparse
import glob
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# Setup path to import fastreid
sys.path.append('fast-reid')

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

setup_logger(name="fastreid")

class PersonDetector:
    def __init__(self, threshold=0.8, device='cuda'):
        self.device = torch.device(device)
        print(f"Loading Person Detector (FasterR-CNN) on {self.device}...")
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.transform = transforms.Compose([transforms.ToTensor()])

    def detect(self, image):
        # image: BGR numpy array
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        boxes = outputs[0]['boxes'].data.cpu().numpy()
        scores = outputs[0]['scores'].data.cpu().numpy()
        labels = outputs[0]['labels'].data.cpu().numpy()
        
        # Keep only people (label 1 for COCO) with high confidence
        mask = (labels == 1) & (scores >= self.threshold)
        return boxes[mask], scores[mask]

class IdentityTracker:
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.targets = [] # List of {'id': N, 'features': feat, 'name': str}
        self.threshold = 0.6  # Cosine similarity threshold
    
    def enroll_target(self, image, name):
        """
        Enroll a target from an image crop.
        """
        # Preprocess logic similar to DefaultPredictor but expects BGR
        # Predictor handles resizing internally if we pass the raw image
        pred = self.predictor(image)
        feat = F.normalize(pred).cpu().data.numpy()
        
        self.targets.append({
            'name': name,
            'features': feat,
            'id': len(self.targets)
        })
        print(f"Enrolled target: {name}")

    def track(self, frame, detections):
        """
        Match detected boxes against enrolled targets.
        """
        matches = []
        for box in detections:
            x1, y1, x2, y2 = map(int, box)
            # Ensure crop is within bounds
            h, w = frame.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2]
            
            # Extract features
            pred = self.predictor(crop)
            feat = F.normalize(pred).cpu().data.numpy()
            
            # Compare with all targets
            best_match = None
            max_score = -1.0
            
            for target in self.targets:
                # Cosine similarity
                score = np.dot(feat, target['features'].T).item()
                if score > max_score:
                    max_score = score
                    best_match = target
            
            if max_score > self.threshold:
                match_info = {
                    'box': (x1, y1, x2, y2),
                    'name': best_match['name'],
                    'score': max_score
                }
                matches.append(match_info)
            else:
                 matches.append({
                    'box': (x1, y1, x2, y2),
                    'name': 'Unknown',
                    'score': max_score
                })
                
        return matches

def setup_cfg(config_file, opts=[]):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Target Enrollment and Tracking Demo")
    parser.add_argument("--config-file", default="custom_configs/bagtricks_R50-ibn.yml", metavar="FILE", help="path to config file")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, or path to video)")
    parser.add_argument("--enroll", help="Path to folder with enrollment images (one per person, filename as name)", required=False)
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    cfg = setup_cfg(args.config_file, args.opts)
    
    detector = PersonDetector(device=cfg.MODEL.DEVICE)
    tracker = IdentityTracker(cfg)

    # Enroll targets if specific path provided
    if args.enroll and os.path.exists(args.enroll):
        valid_exts = ['.jpg', '.png', '.jpeg']
        for file in os.listdir(args.enroll):
            if any(file.lower().endswith(ext) for ext in valid_exts):
                path = os.path.join(args.enroll, file)
                name = os.path.splitext(file)[0]
                img = cv2.imread(path)
                if img is not None:
                    tracker.enroll_target(img, name)
                else:
                    print(f"Failed to read {path}")

    # Open Video Stream
    source = args.source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
        
    print("Starting tracking... Press 'q' to quit.")
    print("Press 'e' to snapshot and enroll the largest detected person (Interactive Enrollment).")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect
        boxes, scores = detector.detect(frame)
        
        # Track/Identify
        results = tracker.track(frame, boxes)
        
        # Draw
        for res in results:
            x1, y1, x2, y2 = res['box']
            name = res['name']
            score = res['score']
            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        cv2.imshow('FastReID Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            # Interactive enrollment
            if len(boxes) > 0:
                # Find largest box
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                max_idx = np.argmax(areas)
                box = boxes[max_idx]
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                name = input("Enter name for this person: ")
                tracker.enroll_target(crop, name)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
