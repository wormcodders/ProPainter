import cv2
import numpy as np
import torch
from ultralytics import SAM

def test_fastsam():
    # Create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1) # white square
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print("Loading model...")
    model = SAM('mobile_sam.pt')
    
    # Predict with point
    print("Running prediction...")
    results = model(img, bboxes=None, points=[200, 200], labels=[1], device=device)
    
    if len(results) > 0 and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()
        print("Mask max:", mask.max())
        print("Mask shape:", mask.shape)
    else:
        print("No mask generated.")
    
if __name__ == "__main__":
    test_fastsam()
