# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

# Workarounds for Blackwell (RTX 50-series) CUBLAS compatibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0).lower()
    major, minor = torch.cuda.get_device_capability(0)
    
    # Fix for CUBLAS_STATUS_INTERNAL_ERROR in torch.matmul (corr.py) on newer architectures
    if major >= 8:  # Ampere (30-series), Ada (40-series), Blackwell (50-series)
        torch.backends.cuda.matmul.allow_tf32 = False
        
    # Nuclear fix for Blackwell CUDNN_STATUS_EXECUTION_FAILED_CUDART in F.conv2d
    if "rtx 50" in device_name or "rtx 40" in device_name or major >= 10:
        torch.backends.cudnn.enabled = False
    else:
        # For Kaggle T4 and older stable GPUs, use full speed cuDNN
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

def log_vram(stage_name=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[VRAM DEBUG] {stage_name} | Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB", flush=True)
import warnings
warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


# resize frames
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size


#  read frames from video
def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        reader = imageio.get_reader(frame_root)
        fps = reader.get_meta_data().get('fps', 24)
        frames = []
        for frame in reader:
            # imageio returns frames in RGB format natively
            frames.append(Image.fromarray(frame))
        reader.close()
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
  
# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
       masks_img = [Image.open(mpath)]
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
          
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))
        mask_img = binary_mask(mask_img).astype(np.uint8)

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = cv2.dilate(mask_img, np.ones((3, 3), np.uint8), iterations=flow_mask_dilates)
        else:
            flow_mask_img = mask_img
            
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img_dilated = cv2.dilate(mask_img, np.ones((3, 3), np.uint8), iterations=mask_dilates)
        else:
            mask_img_dilated = mask_img
            
        masks_dilated.append(Image.fromarray(mask_img_dilated * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []
    
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)
    
    mask[H_start+dilate_h: H_start+imgH-dilate_h, 
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))
  
    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index



if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    print(f"PROPAINTER_DEVICE: {device}", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
    parser.add_argument(
        '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument(
        '--save_frames', action='store_true', help='Save output frames. Default: False')
    parser.add_argument(
        '--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')

    args = parser.parse_args()

    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if args.fp16 else False 
    if device == torch.device('cpu'):
        use_half = False

    print(f"PROPAINTER_STAGE: Reading video frames...", flush=True)
    frames, fps, size, video_name = read_frame_from_videos(args.video)
    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)
    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))

    # Safety cap: auto-downscale only for 4K+ resolutions that physically cannot fit in T4 VRAM
    MAX_SAFE_DIM = 1920
    if args.width == -1 and args.height == -1 and max(size[0], size[1]) > MAX_SAFE_DIM:
        native_w, native_h = size
        sf = MAX_SAFE_DIM / max(native_w, native_h)
        safe_w = max(16, (int(native_w * sf) // 16) * 16)
        safe_h = max(16, (int(native_h * sf) // 16) * 16)
        size = (safe_w, safe_h)
        print(f"PROPAINTER_STAGE: Auto-downscaling from {native_w}x{native_h} to {safe_w}x{safe_h} (4K+ exceeds GPU limits)", flush=True)

    frames, size, out_size = resize_frames(frames, size)
    print(f"PROPAINTER_STAGE: Read {len(frames)} frames ({size[0]}x{size[1]})", flush=True)
    
    fps = args.save_fps if fps is None else fps
    save_root = os.path.join(args.output, video_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    if args.mode == 'video_inpainting':
        frames_len = len(frames)
        print(f"PROPAINTER_STAGE: Reading masks...", flush=True)
        flow_masks, masks_dilated = read_mask(args.mask, frames_len, size, 
                                              flow_mask_dilates=args.mask_dilation,
                                              mask_dilates=args.mask_dilation)
        w, h = size
    elif args.mode == 'video_outpainting':
        assert args.scale_h is not None and args.scale_w is not None, 'Please provide a outpainting scale (s_h, s_w).'
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
        w, h = size
    else:
        raise NotImplementedError
    
    # for saving the masked frames or video
    masked_frame_for_save = []
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_dilated[i]),2).repeat(3, axis=2)/255.
        img = np.array(frames[i])
        green = np.zeros([h, w, 3]) 
        green[:,:,1] = 255
        alpha = 0.6
        # alpha = 1.0
        fuse_img = (1-alpha)*img + alpha*green
        fuse_img = mask_ * fuse_img + (1-mask_)*img
        masked_frame_for_save.append(fuse_img.astype(np.uint8))

    # VRAM Offload Optimization: Use CUDA 1 if available, otherwise System RAM (CPU) for storing massive master tensors
    storage_device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cpu')
    print(f"PROPAINTER_STAGE: Converting to tensors & moving to storage {storage_device}...", flush=True)
    
    # 1. Optimized Tensor Conversion with Aggressive Garbage Collection to avoid OOM Killer (-9)
    from torchvision.transforms.functional import to_tensor
    t_frames = len(frames)
    w, h = size
    
    frames_t = torch.empty((1, t_frames, 3, h, w), dtype=torch.float32, device=storage_device)
    flow_masks_t = torch.empty((1, t_frames, 1, h, w), dtype=torch.float32, device=storage_device)
    masks_dilated_t = torch.empty((1, t_frames, 1, h, w), dtype=torch.float32, device=storage_device)
    frames_inp = []
    
    i = 0
    while frames:
        # Save to frames_inp
        frames_inp.append(np.array(frames[0]).astype(np.uint8))
        
        # Convert to tensor and immediately free the heavy PIL image from System RAM!
        frames_t[0, i] = to_tensor(frames.pop(0)).to(storage_device) * 2.0 - 1.0
        flow_masks_t[0, i] = to_tensor(flow_masks.pop(0)).to(storage_device)
        masks_dilated_t[0, i] = to_tensor(masks_dilated.pop(0)).to(storage_device)
        i += 1
        
    frames = frames_t
    flow_masks = flow_masks_t
    masks_dilated = masks_dilated_t

    
    ##############################################
    # set up RAFT and flow competition model
    ##############################################
    print(f"PROPAINTER_STAGE: Loading RAFT model...", flush=True)
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    print(f"PROPAINTER_STAGE: Loading flow completion model...", flush=True)
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()


    ##############################################
    # set up ProPainter model
    ##############################################
    print(f"PROPAINTER_STAGE: Loading ProPainter model...", flush=True)
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    
    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames.size(1)
    
    # --- GLOBAL VRAM OOM PREVENTION ---
    # Dynamically scale chunk sizes based on resolution
    # 480p (854x480) area is ~410k pixels.
    area = frames.size(-1) * frames.size(-2)
    scale_factor = 410000 / area
    
    # Scale subvideo_length
    args.subvideo_length = max(15, int(args.subvideo_length * scale_factor))
    # Scale neighbor_length (must be even so neighbor_stride works cleanly)
    args.neighbor_length = max(4, int(args.neighbor_length * scale_factor))
    args.neighbor_length = args.neighbor_length - (args.neighbor_length % 2)
    
    if not use_half:
        args.subvideo_length = max(10, args.subvideo_length // 2)
        args.neighbor_length = max(2, args.neighbor_length // 2)
        args.neighbor_length = args.neighbor_length - (args.neighbor_length % 2)
        
    print(f'PROPAINTER_STAGE: Starting inference on {video_length} frames...', flush=True)
    with torch.no_grad():
        # ---- compute flow ----
        print(f"PROPAINTER_STAGE: Computing optical flow (RAFT)...", flush=True)
        log_vram("Pre-RAFT")
        # use fp32 for RAFT
        gt_flows_f_list, gt_flows_b_list = [], []
        
        # Dynamically determine RAFT batch size based on resolution
        if frames.size(-1) <= 640:
            batch_size = 4
        elif frames.size(-1) <= 1280:
            batch_size = 2
        else:
            batch_size = 1 # CRITICAL: 1080p must use batch size 1 to prevent OOM
            
        for f in range(0, video_length - 1, batch_size):
            end_f = min(video_length, f + batch_size + 1)
            flows_f, flows_b = fix_raft(frames[:, f:end_f].to(device), iters=args.raft_iter)
            
            if use_half:
                flows_f, flows_b = flows_f.half(), flows_b.half()
                
            gt_flows_f_list.append(flows_f.to(storage_device))
            gt_flows_b_list.append(flows_b.to(storage_device))
            torch.cuda.empty_cache()
            
        gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
        gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
        gt_flows_bi = (gt_flows_f, gt_flows_b)

        if use_half:
            # Conversion happens safely on the storage device (CPU/CUDA1) to avoid CUDA0 OOM
            frames = frames.half()
            flow_masks = flow_masks.half()
            masks_dilated = masks_dilated.half()
            
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()
            torch.cuda.empty_cache()
        
        # ---- complete flow ----
        print(f"PROPAINTER_STAGE: Completing flow...", flush=True)
        log_vram("Pre-Flow-Completion")
        flow_length = gt_flows_bi[0].size(1)

        if flow_length > args.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, args.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                
                gt_f_sub = gt_flows_bi[0][:, s_f:e_f].to(device)
                gt_b_sub = gt_flows_bi[1][:, s_f:e_f].to(device)
                mask_sub = flow_masks[:, s_f:e_f+1].to(device)
                
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_f_sub, gt_b_sub), mask_sub)
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_f_sub, gt_b_sub), pred_flows_bi_sub, mask_sub)

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e].to(storage_device))
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e].to(storage_device))
                torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            gt_f_sub = gt_flows_bi[0].to(device)
            gt_b_sub = gt_flows_bi[1].to(device)
            mask_sub = flow_masks.to(device)
            
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow((gt_f_sub, gt_b_sub), mask_sub)
            pred_flows_bi = fix_flow_complete.combine_flow((gt_f_sub, gt_b_sub), pred_flows_bi, mask_sub)
            pred_flows_bi = (pred_flows_bi[0].to(storage_device), pred_flows_bi[1].to(storage_device))
            torch.cuda.empty_cache()
            
        # Free massive tensors and models that are no longer needed
        del gt_flows_bi
        del flow_masks
        del fix_raft
        del fix_flow_complete
        torch.cuda.empty_cache()

        # ---- image propagation ----
        print(f"PROPAINTER_STAGE: Image propagation...", flush=True)
        log_vram("Pre-Image-Propagation")
        subvideo_length_img_prop = min(100, args.subvideo_length) # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                # Move chunks to active GPU device
                mask_sub = masks_dilated[:, s_f:e_f].to(device)
                frame_sub = frames[:, s_f:e_f].to(device)
                masked_frame_sub = frame_sub * (1 - mask_sub)
                
                b, t, _, _, _ = mask_sub.size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1].to(device), pred_flows_bi[1][:, s_f:e_f-1].to(device))
                
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                    masked_frame_sub, pred_flows_bi_sub, mask_sub, 'nearest')
                    
                updated_frames_sub = frame_sub * (1 - mask_sub) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * mask_sub
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e].to(storage_device))
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e].to(storage_device))
                torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            mask_sub = masks_dilated.to(device)
            frame_sub = frames.to(device)
            masked_frame_sub = frame_sub * (1 - mask_sub)
            pred_flows_bi_sub = (pred_flows_bi[0].to(device), pred_flows_bi[1].to(device))
            
            b, t, _, _, _ = mask_sub.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frame_sub, pred_flows_bi_sub, mask_sub, 'nearest')
            updated_frames = frame_sub * (1 - mask_sub) + prop_imgs.view(b, t, 3, h, w) * mask_sub
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            updated_frames = updated_frames.to(storage_device)
            updated_masks = updated_masks.to(storage_device)
            torch.cuda.empty_cache()
            
        # Free massive tensors that are no longer needed
        del frames
        torch.cuda.empty_cache()
    
    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = args.neighbor_length // 2
    if video_length > args.subvideo_length:
        ref_num = args.subvideo_length // args.ref_stride
    else:
        ref_num = -1
    
    # ---- feature propagation + transformer ----
    print(f"PROPAINTER_STAGE: Feature propagation + transformer...", flush=True)
    log_vram("Pre-Feature-Propagation")
    for f in range(0, video_length, neighbor_stride):
        print(f"PROPAINTER_PROGRESS: {f} / {video_length}", flush=True)
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :].to(device)
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :].to(device)
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :].to(device)
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :].to(device), pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :].to(device))
        
        with torch.no_grad():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)
            
            # pred_img = selected_imgs # results of image propagation
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            
            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else: 
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        
        # Aggressively free CUDA:0 VRAM between transformer steps
        del selected_imgs, selected_masks, selected_update_masks, selected_pred_flows_bi
        del pred_img
        torch.cuda.empty_cache()
                
    # save each frame
    if args.save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, 'frames', str(idx).zfill(4)+'.png')
            imwrite(f, img_save_root)
                    

    # if args.mode == 'video_outpainting':
    #     comp_frames = [i[10:-10,10:-10] for i in comp_frames]
    #     masked_frame_for_save = [i[10:-10,10:-10] for i in masked_frame_for_save]
    
    # save videos frame
    masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
    imageio.mimwrite(os.path.join(save_root, 'masked_in.mp4'), masked_frame_for_save, fps=fps, quality=7)
    imageio.mimwrite(os.path.join(save_root, 'inpaint_out.mp4'), comp_frames, fps=fps, quality=7)
    
    print(f'\nAll results are saved in {save_root}')
    
    torch.cuda.empty_cache()