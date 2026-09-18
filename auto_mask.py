import cv2
import numpy as np
import torch
import gc
from ultralytics import SAM

def generate_auto_mask(image_np, x, y):
    """
    Given a numpy image (RGB) and a click coordinate (x, y),
    loads MobileSAM, generates a binary mask, unloads the model to free VRAM,
    and returns a visual overlay image and the raw mask.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected! Auto-masking requires a GPU.")
    device = "cuda"

    # Load lightweight MobileSAM
    model = SAM('mobile_sam.pt')
    
    # Run prediction at the clicked point
    results = model(image_np, bboxes=None, points=[x, y], labels=[1], device=device, verbose=False)
    
    if len(results) == 0 or results[0].masks is None:
        # Fallback if nothing found
        return image_np, None
        
    # Get mask data (boolean or float)
    mask_data = results[0].masks.data[0].cpu().numpy()
    
    # Create binary mask (255 for foreground, 0 for background)
    binary_mask = (mask_data * 255).astype(np.uint8)
    
    # Dilate the mask to slightly expand its borders (solves MobileSAM skipping fuzzy watermark edges)
    # 5x5 kernel with 3 iterations adds roughly a 6-pixel buffer around the entire shape
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=3)
    
    # Unload model explicitly to free VRAM for ProPainter
    del model
    del results
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create a visual overlay (red transparent overlay)
    overlay = image_np.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 0] = 255  # Red in RGB
    
    # Alpha blend the mask area
    alpha = 0.5
    mask_bool = binary_mask > 0
    overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 1 - alpha, red_layer[mask_bool], alpha, 0)
    
    # Add a crosshair at the clicked point
    cv2.drawMarker(overlay, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    
    return overlay, binary_mask
