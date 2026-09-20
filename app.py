import gradio as gr
import subprocess
import os
import cv2
import numpy as np
import shutil
import glob
import sys
import re
import torch

# Auto-detect VRAM to set optimal default parameters
try:
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        vram_gb = 0
except:
    vram_gb = 0

# If VRAM > 14GB (e.g. 16GB T4 in Kaggle), push for max quality
if vram_gb >= 14:
    def_res = "Original"
    def_subvid = 80
    def_neighbor = 10
    def_raft_iter = 20
else:
    # 8GB cards (e.g. RTX 5050)
    def_res = "1280x720 (720p)"
    def_subvid = 50
    def_neighbor = 8
    def_raft_iter = 10

def extract_first_frame(videos):
    """Extracts the first frame of the first uploaded video to use as a drawing canvas."""
    if not videos:
        return None, None, None
    vid_path = videos[0].name
    try:
        import imageio
        reader = imageio.get_reader(vid_path)
        frame = reader.get_data(0)  # Gets first frame natively in RGB
        reader.close()
        return frame, frame, None
    except Exception as e:
        print(f"Failed to read first frame with imageio: {e}")
        return None, None, None

def extract_mask_from_editor(mask_dict, mask_path):
    """Extracts the drawing from the ImageEditor layers and saves it as a mask."""
    if not mask_dict or not mask_dict.get("layers"):
        return None
    
    layers = mask_dict["layers"]
    if not layers:
        return None
        
    height, width = layers[0].shape[:2]
    final_mask = np.zeros((height, width), dtype=np.uint8)
    
    drawn = False
    for layer in layers:
        if layer.shape[-1] == 4: # Ensure it has an alpha channel
            alpha = layer[:, :, 3]
            if np.any(alpha > 0):
                drawn = True
            final_mask = np.maximum(final_mask, alpha)
    
    if not drawn:
        return None
        
    # Threshold to make it strictly binary (0 or 255)
    _, final_mask = cv2.threshold(final_mask, 1, 255, cv2.THRESH_BINARY)
    
    is_success, buffer = cv2.imencode(".png", final_mask)
    if is_success:
        with open(mask_path, "wb") as f:
            f.write(buffer)
    else:
        # fallback
        cv2.imwrite(mask_path, final_mask)
        
    return os.path.abspath(mask_path)
import threading

# Define strict execution locks for each GPU to enable smart-queuing
import torch
sys_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
gpu_locks = {str(i): threading.Lock() for i in range(sys_gpu_count)}
gpu_locks["Auto"] = threading.Lock()

# Cache for LaMa image inpainting model to avoid reloading on every image
lama_model = None

os.makedirs("results", exist_ok=True)

# Global registry to track processing status for the Dashboard
registry_lock = threading.Lock()
job_registry = []

def process_videos(videos, masks, mask_editor, auto_mask_state, max_resolution, fp16, raft_iters, subvideo_length, neighbor_length, gpu_select, task_selection, progress=gr.Progress(track_tqdm=False)):
    if not videos:
        yield "Error: No videos uploaded.", []
        return
        
    if not task_selection or len(task_selection) == 0:
        yield "Error: You must select at least one pipeline task.", []
        return

    do_watermark = "Watermark Removal" in task_selection
    do_metadata = "Meta Tag Removal / Forge Apple Metadata" in task_selection

    final_output_paths = []

    gpu_id = gpu_select.replace("GPU ", "").strip() if gpu_select and gpu_select != "Auto" else "Auto"
    if gpu_locks[gpu_id].locked():
        yield f"⏳ Waiting for {gpu_select} to become available... (Queued)\n", []
        
    with gpu_locks[gpu_id]:
        yield f"✅ [{gpu_select} Acquired!] Starting Batch Processing...\n", []

        # Check if user uploaded a mask file
        uploaded_mask_path = None
        if masks and len(masks) > 0:
            if len(masks) == 1:
                uploaded_mask_path = masks[0].name
            else:
                uploaded_mask_path = os.path.dirname(masks[0].name)
                
        # Create a persistent batch folder in 'results' for this run
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.abspath(os.path.join("results", f"batch_{timestamp}"))
        os.makedirs(batch_output_dir, exist_ok=True)
        
        for video in videos:
            vid_path = video.name
            vid_basename = os.path.basename(vid_path)
            vid_name_no_ext, vid_ext = os.path.splitext(vid_basename)
        
            target_output_path = os.path.join(batch_output_dir, f"{vid_name_no_ext}_no_watermark{vid_ext}")
            
            # Register job
            job_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job_id = f"{job_timestamp}_{vid_basename}"
            with registry_lock:
                job_registry.insert(0, {
                    "id": job_id,
                    "timestamp": job_timestamp,
                    "filename": vid_basename,
                    "status": "⏳ Processing...",
                    "output": None
                })

            current_mask_path = uploaded_mask_path
        
            if do_watermark and not current_mask_path:
                yield f"Extracting mask for {vid_basename}...\n", final_output_paths
                unique_mask_name = f"temp_drawn_mask_{vid_name_no_ext}_{gpu_id}.png"
                try:
                    current_mask_path = extract_mask_from_editor(mask_editor, unique_mask_name)
                except Exception as e:
                    pass
                
                # Fallback to AI auto-mask if no manual drawing
                if not current_mask_path and auto_mask_state is not None:
                    cv2.imwrite(unique_mask_name, auto_mask_state)
                    current_mask_path = os.path.abspath(unique_mask_name)
                
            if do_watermark and not current_mask_path:
                with registry_lock:
                    for j in job_registry:
                        if j["id"] == job_id:
                            j["status"] = "❌ Failed (No Mask)"
                yield f"Error: No mask uploaded and no mask drawn for {vid_basename}.\n", final_output_paths
                continue
                
            # --- IMAGE PROCESSING ---
            if vid_ext.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                try:
                    if do_watermark:
                        yield f"Processing static image {vid_basename} via LaMa AI Inpainting...\n", final_output_paths
                        global lama_model
                        if lama_model is None:
                            yield f"Loading LaMa Model for the first time... (Please wait)\n", final_output_paths
                            from simple_lama_inpainting import SimpleLama
                            lama_model = SimpleLama()
                            
                        img = cv2.imread(vid_path)
                        mask_img = cv2.imread(current_mask_path, cv2.IMREAD_GRAYSCALE)
                        if img.shape[:2] != mask_img.shape[:2]:
                            mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        result_pil = lama_model(img_rgb, mask_img)
                        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(target_output_path, result) # This strips metadata natively
                    else:
                        yield f"Skipping watermark removal for {vid_basename} (Processing metadata only)...\n", final_output_paths
                        import shutil
                        shutil.copy(vid_path, target_output_path)
                        
                    if target_output_path.lower().endswith((".jpg", ".jpeg")):
                        try:
                            import piexif
                        except ImportError:
                            subprocess.run([sys.executable, "-m", "pip", "install", "piexif", "-q"])
                            import piexif
                        
                        if do_metadata:
                            try:
                                zeroth_ifd = {
                                    piexif.ImageIFD.Make: b"Apple",
                                    piexif.ImageIFD.Model: b"iPhone 14 Pro",
                                    piexif.ImageIFD.Software: b"17.0.3"
                                }
                                exif_bytes = piexif.dump({"0th": zeroth_ifd, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None})
                                piexif.insert(exif_bytes, target_output_path)
                                msg = f"SUCCESS: Image saved to {target_output_path} (Forged iPhone 14 Pro Metadata)\n"
                            except Exception as meta_e:
                                msg = f"SUCCESS: Image saved to {target_output_path} (Metadata Stripped, but forge failed: {meta_e})\n"
                        else:
                            # User didn't request metadata removal, so restore original if watermark removed
                            if do_watermark:
                                try:
                                    original_exif = piexif.load(vid_path)
                                    exif_bytes = piexif.dump(original_exif)
                                    piexif.insert(exif_bytes, target_output_path)
                                    msg = f"SUCCESS: Image saved to {target_output_path} (Original Metadata Preserved)\n"
                                except Exception:
                                    msg = f"SUCCESS: Image saved to {target_output_path} (No Original Metadata Found)\n"
                            else:
                                msg = f"SUCCESS: Image saved to {target_output_path} (Unchanged)\n"
                    else:
                        strip_msg = " (Metadata Stripped)" if do_watermark else "" # cv2 strips png meta implicitly
                        msg = f"SUCCESS: Image saved to {target_output_path}{strip_msg}\n"
                        
                    final_output_paths.append(target_output_path)
                    yield msg, final_output_paths
                    
                    with registry_lock:
                        for j in job_registry:
                            if j["id"] == job_id:
                                j["status"] = "✅ Completed"
                                j["output"] = target_output_path
                except Exception as e:
                    with registry_lock:
                        for j in job_registry:
                            if j["id"] == job_id:
                                j["status"] = f"❌ Failed ({str(e)})"
                    yield f"Error processing image {vid_basename}: {e}\n", final_output_paths
                
                # Cleanup for images
                try:
                    if current_mask_path and os.path.basename(current_mask_path).startswith("temp_drawn_mask_") and os.path.exists(current_mask_path):
                        os.remove(current_mask_path)
                    # Removed os.remove(vid_path) because Gradio handles temp files
                except:
                    pass
                continue                  
            # --- VIDEO PROCESSING ---
            best_out = None
            log_output = ""
            expected_result_dir = os.path.join("results", vid_name_no_ext)
            
            if do_watermark:
                import imageio_ffmpeg
                import glob
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                
                # 1. FFmpeg Chunking Phase
                segments_dir = os.path.join(expected_result_dir, "segments")
                os.makedirs(segments_dir, exist_ok=True)
                
                log_output += f"--- Slicing {vid_basename} into manageable chunks ---\n"
                yield log_output, final_output_paths
                
                # Calculate resize dimensions ONCE on the original video to prevent metadata errors on chunks
                resize_args = []
                if max_resolution != "Original":
                    try:
                        if "720p" in max_resolution: max_dim = 1280
                        elif "540p" in max_resolution: max_dim = 960
                        elif "480p" in max_resolution: max_dim = 854
                        else: max_dim = 1280
                    
                        import imageio
                        reader = imageio.get_reader(vid_path)
                        meta = reader.get_meta_data()
                        orig_w, orig_h = meta['size']
                        reader.close()
                    
                        if orig_w > 0 and orig_h > 0:
                            scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
                            new_w = max(16, (int(orig_w * scale) // 16) * 16)
                            new_h = max(16, (int(orig_h * scale) // 16) * 16)
                            resize_args = ["--width", str(new_w), "--height", str(new_h)]
                            log_output += f"Calculated resize resolution: {new_w}x{new_h}\n"
                    except Exception as e:
                        log_output += f"Warning: Failed to calculate proportional resolution: {e}\n"
                
                chunk_pattern = os.path.join(segments_dir, "chunk_%04d.mp4")
                split_cmd = [
                    ffmpeg_exe, "-y", "-i", vid_path,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "18",
                    "-g", "30",
                    "-an",
                    "-f", "segment",
                    "-segment_time", "2",
                    "-reset_timestamps", "1",
                    chunk_pattern
                ]
                subprocess.run(split_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                chunks = sorted(glob.glob(os.path.join(segments_dir, "chunk_*.mp4")))
                if not chunks:
                    chunks = [vid_path]
                
                processed_chunks_dict = {}
                import queue
                import threading
                
                def enqueue_output(out, q, c_idx, g_id):
                    for r_line in out:
                        q.put((c_idx, g_id, r_line))
                    out.close()
                    q.put((c_idx, g_id, None))
                    
                parallel_workers = sys_gpu_count if (sys_gpu_count > 1 and gpu_select == "Auto") else 1
                if parallel_workers > 1:
                    log_output += f"\n--- Parallel Mode: Distributing {len(chunks)} chunks across {parallel_workers} GPUs ---\n"
                else:
                    log_output += f"\n--- Processing {len(chunks)} chunks sequentially ---\n"
                yield log_output, final_output_paths
                
                active_processes = []
                q = queue.Queue()
                threads = []
                chunk_idx = 0
                error_occurred = False
                
                while chunk_idx < len(chunks) or active_processes:
                    while len(active_processes) < parallel_workers and chunk_idx < len(chunks) and not error_occurred:
                        chunk_path = chunks[chunk_idx]
                        
                        cmd = [
                            sys.executable, "-u", "inference_propainter.py",
                            "--video", chunk_path,
                            "--mask", current_mask_path,
                            "--raft_iter", str(int(raft_iters)),
                            "--subvideo_length", str(int(subvideo_length)),
                            "--neighbor_length", str(int(neighbor_length))
                        ]
                        if resize_args: cmd.extend(resize_args)
                        if fp16: cmd.append("--fp16")
                    
                        env = os.environ.copy()
                        env["PYTHONUNBUFFERED"] = "1"
                        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                        
                        if parallel_workers > 1:
                            gpu_to_use = str(chunk_idx % parallel_workers)
                            env["CUDA_VISIBLE_DEVICES"] = gpu_to_use
                        elif gpu_select and gpu_select != "Auto":
                            env["CUDA_VISIBLE_DEVICES"] = gpu_select.replace("GPU ", "").strip()
                        else:
                            gpu_to_use = "0"
                            
                        process = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0
                        )
                        
                        g_id_display = env.get("CUDA_VISIBLE_DEVICES", "0")
                        t = threading.Thread(target=enqueue_output, args=(process.stdout, q, chunk_idx, g_id_display))
                        t.daemon = True
                        t.start()
                        threads.append(t)
                        
                        active_processes.append({
                            "process": process,
                            "chunk_idx": chunk_idx,
                            "chunk_path": chunk_path,
                            "gpu": g_id_display
                        })
                        
                        log_output += f">> Started Chunk {chunk_idx+1}/{len(chunks)} on GPU {g_id_display}\n"
                        yield log_output, final_output_paths
                        
                        chunk_idx += 1
                        
                    try:
                        c_idx, g_id, raw_line = q.get(timeout=0.2)
                        if raw_line is not None:
                            line = raw_line.decode("utf-8", errors="replace")
                            prefix = f"[Chunk {c_idx+1}] " if parallel_workers > 1 else ""
                            
                            if "PROPAINTER_DEVICE:" in line:
                                dev_display = line.strip().split(":")[-1].strip().upper()
                                log_output += f"{prefix}DEVICE: {dev_display}\n"
                                yield log_output, final_output_paths
                            elif "PROPAINTER_STAGE:" in line:
                                log_output += f"{prefix}>> {line.strip().split(':', 1)[-1].strip()}\n"
                                yield log_output, final_output_paths
                            elif "PROPAINTER_PROGRESS:" in line:
                                try:
                                    parts = line.strip().split(":")[-1].split("/")
                                    current, total = int(parts[0].strip()), int(parts[1].strip())
                                    pct = current / total
                                    pct_int = int(pct * 100)
                                    
                                    completed_chunks = len(processed_chunks_dict)
                                    overall_pct = ((completed_chunks + (pct / len(active_processes))) / len(chunks)) * 100
                                    progress(overall_pct/100, desc=f"ProPainter Inference ({completed_chunks}/{len(chunks)} chunks done)")
                                    
                                    if pct_int % 10 == 0 or pct_int == 100:
                                        log_output += f"{prefix}Progress: {pct_int}% ({current}/{total})\n"
                                        yield log_output, final_output_paths
                                except Exception:
                                    pass
                            else:
                                log_output += f"{prefix}{line}"
                                yield log_output, final_output_paths
                    except queue.Empty:
                        pass
                        
                    still_active = []
                    for p_info in active_processes:
                        p = p_info["process"]
                        c_idx = p_info["chunk_idx"]
                        c_path = p_info["chunk_path"]
                        
                        if p.poll() is not None:
                            if p.returncode != 0:
                                log_output += f"\nError: ProPainter exited with code {p.returncode} on chunk {c_idx+1}.\n"
                                yield log_output, final_output_paths
                                error_occurred = True
                            else:
                                chunk_name = os.path.splitext(os.path.basename(c_path))[0]
                                chunk_result_dir = os.path.join("results", chunk_name)
                                chunk_out_path = os.path.join(segments_dir, f"processed_{c_idx:04d}.mp4")
                                default_out = os.path.join(chunk_result_dir, "inpaint_out.mp4")
                                if os.path.exists(default_out):
                                    import shutil
                                    shutil.move(default_out, chunk_out_path)
                                    processed_chunks_dict[c_idx] = chunk_out_path
                                    try: shutil.rmtree(chunk_result_dir)
                                    except: pass
                                log_output += f">> Finished Chunk {c_idx+1}/{len(chunks)}!\n"
                                yield log_output, final_output_paths
                        else:
                            still_active.append(p_info)
                            
                    active_processes = still_active
                    
                    if error_occurred and not active_processes:
                        break
                        
                if error_occurred or len(processed_chunks_dict) < len(chunks):
                    with registry_lock:
                        for j in job_registry:
                            if j["id"] == job_id:
                                j["status"] = "❌ Failed (ProPainter Error)"
                    continue
                    
                processed_chunks = [processed_chunks_dict[i] for i in range(len(chunks))]
                
                log_output += f"\n--- Stitching {len(processed_chunks)} chunks back together ---\n"
                yield log_output, final_output_paths
                
                concat_list = os.path.join(segments_dir, "concat.txt")
                with open(concat_list, "w") as f:
                    for c in processed_chunks:
                        f.write(f"file '{os.path.abspath(c).replace(chr(92), '/')}'\n")
                        
                merged_out = os.path.join(expected_result_dir, "merged_inpaint_out.mp4")
                concat_cmd = [
                    ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_list,
                    "-c", "copy",
                    merged_out
                ]
                subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                best_out = merged_out if os.path.exists(merged_out) else None
                if not best_out:
                    log_output += "\nError: Failed to merge processed chunks.\n"
                    yield log_output, final_output_paths
                    continue
            else:
                log_output = f"--- Skipping watermark removal for {vid_basename} (Processing metadata only) ---\n"
                best_out = vid_path
                yield log_output, final_output_paths

            output_found = False
            if best_out:
                try:
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    
                    ff_cmd = [ffmpeg_exe, "-y", "-i", best_out]
                    
                    if do_watermark:
                        ff_cmd.extend([
                            "-i", vid_path,
                            "-c:v", "copy",
                            "-c:a", "aac",
                            "-map", "0:v:0",
                            "-map", "1:a:0?"
                        ])
                    else:
                        ff_cmd.extend(["-c", "copy"])
                        
                    if do_metadata:
                        ff_cmd.extend([
                            "-map_metadata", "-1",
                            "-metadata", "make=Apple",
                            "-metadata", "model=iPhone 14 Pro",
                            "-metadata", "software=17.0.3",
                            "-metadata:s:v:0", "handler_name=Core Media Video",
                            "-metadata:s:a:0", "handler_name=Core Media Audio"
                        ])
                    else:
                        ff_cmd.extend(["-map_metadata", "0"])

                    ff_cmd.append(target_output_path)
                
                    ff_proc = subprocess.run(ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if ff_proc.returncode == 0 and os.path.exists(target_output_path):
                        output_found = True
                    else:
                        err = ff_proc.stderr.decode('utf-8', errors='ignore')
                        log_output += f"Warning: ffmpeg process failed (code {ff_proc.returncode}):\n{err}\n"
                        # Fallback copy
                        import shutil
                        shutil.copy(best_out, target_output_path)
                        output_found = True
                except Exception as e:
                    log_output += f"Warning: Failed to process video with ffmpeg: {e}\n"
                    import shutil
                    shutil.copy(best_out, target_output_path)
                    output_found = True                  
        

        
            if output_found:
                final_output_paths.append(target_output_path)
                log_output += f"\nSUCCESS: Output saved to {target_output_path}\n"
                with registry_lock:
                    for j in job_registry:
                        if j["id"] == job_id:
                            j["status"] = "✅ Completed"
                            j["output"] = target_output_path
            else:
                log_output += f"\nWARNING: Could not locate output video for {vid_basename}. Check terminal for details.\n"
                with registry_lock:
                    for j in job_registry:
                        if j["id"] == job_id:
                            j["status"] = "❌ Failed"
            
            # --- CLEANUP TEMP FILES ---
            try:
                if os.path.exists(expected_result_dir):
                    shutil.rmtree(expected_result_dir)
                if current_mask_path and os.path.basename(current_mask_path).startswith("temp_drawn_mask_") and os.path.exists(current_mask_path):
                    os.remove(current_mask_path)
                # Removed os.remove(vid_path) because Gradio handles temp files and identical uploads share the same path
                log_output += "Cleaned up temporary files.\n"
            except Exception as e:
                log_output += f"Warning: Failed to clean up temp files: {e}\n"
            
            yield log_output, final_output_paths

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
gpu_choices = ["Auto"] + [f"GPU {i}" for i in range(gpu_count)] if gpu_count > 0 else ["Auto"]

with gr.Blocks(title="ProPainter Local GUI") as demo:
    gr.Markdown("# Local ProPainter Web UI")
    gr.Markdown("Lightweight, memory-efficient UI for removing watermarks using ProPainter. Optimized for 8GB VRAM RTX GPUs.")
    
    with gr.Tabs():
        with gr.Tab('Processor'):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Select Tasks to Perform")
                    task_selection = gr.CheckboxGroup(
                        choices=["Watermark Removal", "Meta Tag Removal / Forge Apple Metadata"],
                        value=["Watermark Removal", "Meta Tag Removal / Forge Apple Metadata"],
                        label="Pipeline Tasks"
                    )
                    
                    gr.Markdown("### 2. Upload Media (Video or Image)")
                    video_input = gr.File(file_count="multiple", label="Upload Files (MP4, AVI, MOV, JPG, PNG)", file_types=["video", "image"])
                    
                    gr.Markdown("### 2. Mask the Watermark")
                    gr.Markdown("When you upload a video, its first frame will appear below.")
                    
                    with gr.Tabs():
                        with gr.TabItem("Auto-Select (Click)"):
                            gr.Markdown("Click **exactly on the watermark** to automatically outline it using AI (MobileSAM).")
                            auto_mask_image = gr.Image(label="Click to Auto-Mask", type="numpy", interactive=True)
                            auto_mask_state = gr.State(None)
                            clean_frame_state = gr.State(None)
                            
                        with gr.TabItem("Manual Draw"):
                            gr.Markdown("Use the brush tool to paint over the watermark.")
                            mask_editor = gr.ImageEditor(
                                label="Draw Mask",
                                type="numpy",
                                interactive=True,
                                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                                eraser=gr.Eraser()
                            )
                    
                    reset_drawing_btn = gr.Button("Reset Mask Canvas", size="sm")
                    
                    # Automatically update the canvases when a video is uploaded
                    video_input.change(
                        fn=lambda v: (lambda f: (f, f, None, f))(extract_first_frame(v)[0]),
                        inputs=video_input, 
                        outputs=[mask_editor, auto_mask_image, auto_mask_state, clean_frame_state]
                    )
                    
                    # Allow user to explicitly clear the drawing
                    reset_drawing_btn.click(
                        fn=lambda v: (lambda f: (f, f, None, f))(extract_first_frame(v)[0]),
                        inputs=video_input,
                        outputs=[mask_editor, auto_mask_image, auto_mask_state, clean_frame_state]
                    )
                    
                    # Handle click on Auto-Mask image
                    def handle_auto_mask_click(evt: gr.SelectData, clean_frame, current_mask_state):
                        if clean_frame is None:
                            return None, None
                        from auto_mask import generate_auto_mask
                        # Run SAM on the clean frame so previous red overlays don't confuse the model
                        _, new_mask = generate_auto_mask(clean_frame, evt.index[0], evt.index[1])
                        
                        # Accumulate multiple watermark clicks
                        if current_mask_state is not None:
                            merged_mask = cv2.bitwise_or(current_mask_state, new_mask)
                        else:
                            merged_mask = new_mask
                            
                        # Generate the combined visual overlay
                        final_overlay = clean_frame.copy()
                        red_layer = np.zeros_like(final_overlay)
                        red_layer[:, :, 0] = 255
                        alpha = 0.5
                        mask_bool = merged_mask > 0
                        final_overlay[mask_bool] = cv2.addWeighted(final_overlay[mask_bool], 1 - alpha, red_layer[mask_bool], alpha, 0)
                        cv2.drawMarker(final_overlay, (evt.index[0], evt.index[1]), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                        
                        return final_overlay, merged_mask
        
                    auto_mask_image.select(
                        fn=handle_auto_mask_click,
                        inputs=[clean_frame_state, auto_mask_state],
                        outputs=[auto_mask_image, auto_mask_state]
                    )
                    
                    gr.Markdown("### Or Upload External Mask (Optional)")
                    mask_input = gr.File(file_count="multiple", label="Upload Mask (PNG) or Sequential Mask Frames", file_types=["image"])
                        
                    gr.Markdown("### 3. VRAM Optimization Settings (Auto-Configured for your GPU)")
                    
                    with gr.Row():
                        gpu_select = gr.Dropdown(choices=gpu_choices, value="Auto", label="Target GPU (For Multi-GPU setups)")
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        max_resolution = gr.Dropdown(
                            choices=["Original", "1280x720 (720p)", "960x540 (540p)", "854x480 (480p)"], 
                            value=def_res, 
                            label=f"Processing Resolution (VRAM Auto-Detected: ~{int(vram_gb)}GB)"
                        )
                        fp16_opt = gr.Checkbox(label="Enable FP16 (Half Precision)", value=True)
                        raft_iters = gr.Slider(minimum=5, maximum=20, step=1, label="RAFT Iterations (Lower = Much Faster, slightly worse flow)", value=def_raft_iter)
                        subvid_len = gr.Slider(minimum=10, maximum=100, step=5, label="Subvideo Length", value=def_subvid)
                        neighbor_len = gr.Slider(minimum=2, maximum=16, step=2, label="Neighbor Length", value=def_neighbor)
                    
                    process_btn = gr.Button("Process Video(s)", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Progress & Console Output")
                    console_output = gr.Textbox(label="Status / Logs", lines=20, max_lines=30, interactive=False)
                    
                    gr.Markdown("### Finished Videos")
                    file_outputs = gr.File(label="Processed Videos", interactive=False)
                    
    with gr.Tab("Results Dashboard"):
        gr.Markdown("### 📊 Live Processing Dashboard")
        gr.Markdown("View the status of all queued and completed media. Click the link to download finished files directly.")
        
        dashboard_html = gr.HTML()
        
        def render_dashboard():
            with registry_lock:
                if not job_registry:
                    return "<div style='padding: 20px; text-align: center; color: gray;'>No jobs have been submitted yet.</div>"
                
                html_content = "<table style='width: 100%; text-align: left; border-collapse: collapse;'>"
                html_content += "<tr style='border-bottom: 2px solid #ddd;'><th>Timestamp</th><th>File</th><th>Status</th><th>Download</th></tr>"
                for j in job_registry:
                    status_color = "black"
                    if "Completed" in j['status']: status_color = "green"
                    elif "Failed" in j['status']: status_color = "red"
                    elif "Processing" in j['status']: status_color = "blue"
                    
                    if j['output']:
                        import urllib.parse
                        import html
                        safe_path = str(j['output']).replace('\\', '/')
                        safe_url = f"/file={urllib.parse.quote(safe_path, safe='/:')}"
                        
                        # Get the actual output filename, not just the original input filename
                        # This ensures the browser saves it as _no_watermark.jpeg instead of .customization
                        actual_filename = os.path.basename(j['output'])
                        safe_filename = html.escape(actual_filename, quote=True)
                        
                        dl_link = f"<a href='{safe_url}' target='_blank' download='{safe_filename}'>📥 Download</a>"
                    else:
                        dl_link = "-"
                    
                    html_content += f"<tr style='border-bottom: 1px solid #eee; height: 40px;'>"
                    html_content += f"<td>{j['timestamp']}</td>"
                    html_content += f"<td>{j['filename']}</td>"
                    html_content += f"<td style='color: {status_color}; font-weight: bold;'>{j['status']}</td>"
                    html_content += f"<td>{dl_link}</td>"
                    html_content += "</tr>"
                html_content += "</table>"
                return html_content

        refresh_btn = gr.Button("🔄 Refresh Dashboard")
        refresh_btn.click(fn=render_dashboard, inputs=[], outputs=[dashboard_html])
        demo.load(fn=render_dashboard, inputs=[], outputs=[dashboard_html])
        
    def update_mask_visibility(tasks):
        is_active = "Watermark Removal" in (tasks or [])
        return [
            gr.update(interactive=is_active),
            gr.update(interactive=is_active),
            gr.update(interactive=is_active)
        ]
        
    task_selection.change(
        fn=update_mask_visibility,
        inputs=[task_selection],
        outputs=[auto_mask_image, mask_editor, mask_input]
    )

    process_btn.click(
        fn=process_videos,
        inputs=[video_input, mask_input, mask_editor, auto_mask_state, max_resolution, fp16_opt, raft_iters, subvid_len, neighbor_len, gpu_select, task_selection],
        outputs=[console_output, file_outputs]
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ProPainter Local Web UI")
    parser.add_argument("--cloudflare", action="store_true", help="Start a Cloudflare tunnel for a public URL (Useful for Colab/Kaggle)")
    parser.add_argument("--share", action="store_true", help="Use standard Gradio share=True")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    args = parser.parse_args()

    if args.cloudflare:
        import platform, subprocess, threading, urllib.request, stat
        def start_cloudflare_tunnel(port):
            print("Starting Cloudflare tunnel...")
            system = platform.system().lower()
            if system == "linux":
                bin_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
                bin_path = "./cloudflared"
            elif system == "windows":
                bin_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
                bin_path = "./cloudflared.exe"
            else:
                print("Cloudflare tunnel only auto-supported on Linux/Windows.")
                return

            if not os.path.exists(bin_path):
                print(f"Downloading cloudflared to {bin_path}...")
                urllib.request.urlretrieve(bin_url, bin_path)
                if system == "linux":
                    st = os.stat(bin_path)
                    os.chmod(bin_path, st.st_mode | stat.S_IEXEC)
                    
            def run_tunnel():
                process = subprocess.Popen(
                    [bin_path, "tunnel", "--url", f"http://127.0.0.1:{port}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                for line in process.stdout:
                    if "trycloudflare.com" in line:
                        url = re.search(r'https://[-a-z0-9]+\.trycloudflare\.com', line)
                        if url:
                            print(f"\n{'='*60}")
                            print(f"🌍 CLOUDFLARE PUBLIC URL: {url.group(0)}")
                            print(f"{'='*60}\n")
            threading.Thread(target=run_tunnel, daemon=True).start()
            
        start_cloudflare_tunnel(args.port)

    demo.queue(default_concurrency_limit=2)  # Allow up to 2 parallel processing jobs across tabs
    # On Kaggle/Colab, we want to listen on all interfaces (0.0.0.0) so it's accessible.
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, inbrowser=(not args.cloudflare and not args.share), allowed_paths=[os.path.abspath("results")])
