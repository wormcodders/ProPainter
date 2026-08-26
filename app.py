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
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None, None
    # Convert BGR to RGB for Gradio Image component
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, frame, None

def extract_mask_from_editor(mask_dict, video_path):
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
    
    mask_path = "temp_drawn_mask.png"
    is_success, buffer = cv2.imencode(".png", final_mask)
    if is_success:
        with open(mask_path, "wb") as f:
            f.write(buffer)
    else:
        # fallback
        cv2.imwrite(mask_path, final_mask)
        
    return os.path.abspath(mask_path)

def process_videos(videos, masks, mask_editor, auto_mask_state, max_resolution, fp16, raft_iters, subvideo_length, neighbor_length, progress=gr.Progress(track_tqdm=False)):
    if not videos:
        yield "Error: No videos uploaded.", []
        return

    final_output_paths = []

    # Check if user uploaded a mask file
    uploaded_mask_path = None
    if masks and len(masks) > 0:
        if len(masks) == 1:
            uploaded_mask_path = masks[0].name
        else:
            uploaded_mask_path = os.path.dirname(masks[0].name)
    
    for video in videos:
        vid_path = video.name
        vid_dir = os.path.dirname(vid_path)
        vid_basename = os.path.basename(vid_path)
        vid_name_no_ext, vid_ext = os.path.splitext(vid_basename)
        
        target_output_path = os.path.join(vid_dir, f"{vid_name_no_ext}_no_watermark{vid_ext}")

        current_mask_path = uploaded_mask_path
        
        if not current_mask_path:
            yield f"Extracting mask for {vid_basename}...\n", final_output_paths
            try:
                current_mask_path = extract_mask_from_editor(mask_editor, vid_path)
            except Exception as e:
                pass
                
            # Fallback to AI auto-mask if no manual drawing
            if not current_mask_path and auto_mask_state is not None:
                mask_path = "temp_drawn_mask.png"
                cv2.imwrite(mask_path, auto_mask_state)
                current_mask_path = os.path.abspath(mask_path)
                
        if not current_mask_path:
            yield f"Error: No mask uploaded and no mask drawn for {vid_basename}.\n", final_output_paths
            continue

        cmd = [
            sys.executable, "-u", "inference_propainter.py",
            "--video", vid_path,
            "--mask", current_mask_path,
            "--raft_iter", str(int(raft_iters)),
            "--subvideo_length", str(int(subvideo_length)),
            "--neighbor_length", str(int(neighbor_length))
        ]
        
        if max_resolution != "Original":
            try:
                res_part = max_resolution.split(" ")[0]
                w, h = res_part.split("x")
                cmd.extend(["--width", w, "--height", h])
            except Exception:
                pass

        if fp16:
            cmd.append("--fp16")

        yield f"Starting processing for {vid_basename}...\nCommand: {' '.join(cmd)}\n", final_output_paths
        
        # Set PYTHONUNBUFFERED so the child process flushes stdout immediately
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0  # unbuffered binary mode
        )

        log_output = f"--- Processing {vid_basename} ---\n"
        device_reported = False
        last_pct = -1
        
        # Read line-by-line from the raw binary stream to avoid text-mode buffering
        for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            
            if "PROPAINTER_DEVICE:" in line:
                dev = line.strip().split(":")[-1].strip()
                dev_display = dev.upper()
                log_output += f"\n{'='*50}\n"
                log_output += f"  DEVICE: {dev_display}\n"
                log_output += f"{'='*50}\n\n"
                device_reported = True
                yield log_output, final_output_paths
            
            elif "PROPAINTER_STAGE:" in line:
                stage = line.strip().split(":", 1)[-1].strip()
                log_output += f">> {stage}\n"
                yield log_output, final_output_paths
                
            elif "PROPAINTER_PROGRESS:" in line:
                try:
                    parts = line.strip().split(":")[-1].split("/")
                    current = int(parts[0].strip())
                    total = int(parts[1].strip())
                    pct = current / total
                    pct_int = int(pct * 100)
                    
                    # Update Gradio progress bar
                    progress(pct, desc=f"ProPainter Inference ({pct_int}%)")
                    
                    # Also update the log textbox every 5%
                    if pct_int >= last_pct + 5 or pct_int == 0:
                        last_pct = pct_int
                        bar_len = 30
                        filled = int(bar_len * pct)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        log_output += f"  Progress: [{bar}] {pct_int}%  ({current}/{total} steps)\n"
                        yield log_output, final_output_paths
                except Exception:
                    log_output += line
                    yield log_output, final_output_paths
            else:
                log_output += line
                yield log_output, final_output_paths

        process.wait()
        
        if process.returncode != 0:
            log_output += f"\nError: ProPainter exited with code {process.returncode} for {vid_basename}.\n"
            yield log_output, final_output_paths
            continue

        # Mark 100%
        progress(1.0, desc="ProPainter Inference (100%)")
        bar = "█" * 30
        log_output += f"  Progress: [{bar}] 100%  (Complete)\n"

        # Find and move output
        expected_result_dir = os.path.join("results", vid_name_no_ext)
        output_found = False
        
        if os.path.exists(expected_result_dir):
            out_files = glob.glob(os.path.join(expected_result_dir, "*.mp4"))
            if out_files:
                out_files.sort(key=os.path.getmtime, reverse=True)
                best_out = out_files[0]
                shutil.move(best_out, target_output_path)
                output_found = True
        
        if not output_found:
            alt_out = os.path.join("results", f"{vid_name_no_ext}.mp4")
            if os.path.exists(alt_out):
                shutil.move(alt_out, target_output_path)
                output_found = True
        
        if output_found:
            final_output_paths.append(target_output_path)
            log_output += f"\nSUCCESS: Output saved to {target_output_path}\n"
        else:
            log_output += f"\nWARNING: Could not locate output video for {vid_basename}. Check terminal for details.\n"
            
        # --- CLEANUP TEMP FILES ---
        try:
            if os.path.exists(expected_result_dir):
                shutil.rmtree(expected_result_dir)
            if current_mask_path and os.path.basename(current_mask_path) == "temp_drawn_mask.png" and os.path.exists(current_mask_path):
                os.remove(current_mask_path)
            log_output += "Cleaned up temporary files.\n"
        except Exception as e:
            log_output += f"Warning: Failed to clean up temp files: {e}\n"
            
        yield log_output, final_output_paths

with gr.Blocks(title="ProPainter Local GUI") as demo:
    gr.Markdown("# Local ProPainter Web UI")
    gr.Markdown("Lightweight, memory-efficient UI for removing watermarks using ProPainter. Optimized for 8GB VRAM RTX GPUs.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Video(s)")
            video_input = gr.File(file_count="multiple", label="Upload Video Files (MP4, AVI, MOV)", file_types=["video"])
            
            gr.Markdown("### 2. Mask the Watermark")
            gr.Markdown("When you upload a video, its first frame will appear below.")
            
            with gr.Tabs():
                with gr.TabItem("Auto-Select (Click)"):
                    gr.Markdown("Click **exactly on the watermark** to automatically outline it using AI (MobileSAM).")
                    auto_mask_image = gr.Image(label="Click to Auto-Mask", type="numpy", interactive=True)
                    auto_mask_state = gr.State(None)
                    
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
                fn=extract_first_frame, 
                inputs=video_input, 
                outputs=[mask_editor, auto_mask_image, auto_mask_state]
            )
            
            # Allow user to explicitly clear the drawing
            reset_drawing_btn.click(
                fn=extract_first_frame,
                inputs=video_input,
                outputs=[mask_editor, auto_mask_image, auto_mask_state]
            )
            
            # Handle click on Auto-Mask image
            def handle_auto_mask_click(evt: gr.SelectData, frame):
                if frame is None:
                    return frame, None
                from auto_mask import generate_auto_mask
                overlay, mask = generate_auto_mask(frame, evt.index[0], evt.index[1])
                return overlay, mask

            auto_mask_image.select(
                fn=handle_auto_mask_click,
                inputs=[auto_mask_image],
                outputs=[auto_mask_image, auto_mask_state]
            )
            
            gr.Markdown("### Or Upload External Mask (Optional)")
            mask_input = gr.File(file_count="multiple", label="Upload Mask (PNG) or Sequential Mask Frames", file_types=["image"])
                
            gr.Markdown("### 3. VRAM Optimization Settings (Auto-Configured for your GPU)")
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
            
    process_btn.click(
        fn=process_videos,
        inputs=[video_input, mask_input, mask_editor, auto_mask_state, max_resolution, fp16_opt, raft_iters, subvid_len, neighbor_len],
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

    demo.queue()  # Required for gr.Progress() to work
    # On Kaggle/Colab, we want to listen on all interfaces (0.0.0.0) so it's accessible.
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, inbrowser=(not args.cloudflare and not args.share))
