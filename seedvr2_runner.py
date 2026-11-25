import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable, Tuple, Generator

# Base directories
BASE_DIR = Path(__file__).resolve().parent
SEEDVR2_DIR = BASE_DIR / "seedvr2"
CLI_SCRIPT = SEEDVR2_DIR / "inference_cli.py"
MODEL_DIR = SEEDVR2_DIR / "models" / "SEEDVR2"

def get_python_exe() -> str:
    """Returns the dedicated venv python executable if present, else fallback to sys.executable."""
    venv_py = BASE_DIR / "venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable

REPO_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
HF_REPO_ID = "numz/SeedVR2_comfyUI"
DEFAULT_DIT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_VAE_MODEL = "ema_vae_fp16.safetensors"


def patch_inference_cli(cli_file: Path, log_callback: Optional[Callable[[str], None]] = None):
    """Patches single-image multi-GPU handling in inference_cli.py if needed."""
    if not cli_file.exists():
        return
    try:
        with open(cli_file, "r", encoding="utf-8") as f:
            content = f.read()

        target = "if len(device_list) > 1:"
        replacement = "if len(device_list) > 1 and frames_tensor.shape[0] > 1:"
        if target in content and replacement not in content:
            content = content.replace(target, replacement, 1)
            with open(cli_file, "w", encoding="utf-8") as f:
                f.write(content)
            if log_callback:
                log_callback("Applied single-image multi-GPU patch to inference_cli.py\n")
    except Exception as e:
        if log_callback:
            log_callback(f"Warning: Failed to patch inference_cli.py: {e}\n")


def ensure_seedvr2_models(log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Verifies that the required VAE and DiT weights exist in models/SEEDVR2, downloading them if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vae_path = MODEL_DIR / DEFAULT_VAE_MODEL
    dit_path = MODEL_DIR / DEFAULT_DIT_MODEL

    from huggingface_hub import hf_hub_download

    # 1. Download VAE if missing (~335MB)
    if not vae_path.exists():
        msg = f"Downloading SeedVR2 VAE model ({DEFAULT_VAE_MODEL})...\n"
        if log_callback:
            log_callback(msg)
        else:
            print(msg, flush=True)
        try:
            hf_hub_download(repo_id=HF_REPO_ID, filename=DEFAULT_VAE_MODEL, local_dir=str(MODEL_DIR))
            msg = f"✅ VAE model downloaded successfully to {vae_path}\n"
            if log_callback:
                log_callback(msg)
            else:
                print(msg, flush=True)
        except Exception as e:
            err = f"❌ Failed to download VAE model: {e}\n"
            if log_callback:
                log_callback(err)
            else:
                print(err, flush=True)
            return False

    # 2. Download DiT if missing (~3.5GB for FP8)
    if not dit_path.exists():
        msg = f"Downloading SeedVR2 DiT model ({DEFAULT_DIT_MODEL}, ~3.5GB)... Please wait.\n"
        if log_callback:
            log_callback(msg)
        else:
            print(msg, flush=True)
        try:
            hf_hub_download(repo_id=HF_REPO_ID, filename=DEFAULT_DIT_MODEL, local_dir=str(MODEL_DIR))
            msg = f"✅ DiT model downloaded successfully to {dit_path}\n"
            if log_callback:
                log_callback(msg)
            else:
                print(msg, flush=True)
        except Exception as e:
            err = f"❌ Failed to download DiT model: {e}\n"
            if log_callback:
                log_callback(err)
            else:
                print(err, flush=True)
            return False

    return True


def setup_seedvr2(log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Ensures repository is cloned, patched, and models are available."""
    if not SEEDVR2_DIR.exists() or not CLI_SCRIPT.exists():
        msg = f"Cloning SeedVR2 repository from {REPO_URL}...\n"
        if log_callback:
            log_callback(msg)
        else:
            print(msg, flush=True)
        try:
            subprocess.run(["git", "clone", REPO_URL, str(SEEDVR2_DIR)], check=True, capture_output=True, text=True)
            msg = f"✅ SeedVR2 cloned to {SEEDVR2_DIR}\n"
            if log_callback:
                log_callback(msg)
            else:
                print(msg, flush=True)
        except Exception as e:
            err = f"❌ Failed to clone SeedVR2: {e}\n"
            if log_callback:
                log_callback(err)
            else:
                print(err, flush=True)
            return False

    patch_inference_cli(CLI_SCRIPT, log_callback)
    return ensure_seedvr2_models(log_callback)


def stream_seedvr2_upscale(
    input_path: str,
    output_path: str,
    is_video: bool,
    resolution: int = 1080,
    gpu_select: str = "Auto",
    chunk_size: int = 45,
    temporal_overlap: int = 2,
    batch_size: int = 5,
    color_correction: str = "lab",
    vae_tiled: bool = True,
    tile_size: int = 1024,
    dit_model: str = DEFAULT_DIT_MODEL,
    blocks_to_swap: int = 0
) -> Generator[Tuple[str, bool, bool, str], None, None]:
    """
    Executes inference_cli.py yielding real-time output chunks.
    
    Yields:
        (log_line: str, finished: bool, success: bool, result_path_or_error: str)
    """
    if not os.path.exists(input_path):
        yield f"Input file not found: {input_path}\n", True, False, f"Input file not found: {input_path}"
        return

    # Ensure repository and weights are ready
    if not CLI_SCRIPT.exists() or not (MODEL_DIR / dit_model).exists() or not (MODEL_DIR / DEFAULT_VAE_MODEL).exists():
        yield "Checking/downloading SeedVR2 components before starting...\n", False, False, ""
        if not setup_seedvr2(lambda m: None):
            yield "Failed to initialize SeedVR2 engine or download required weights.\n", True, False, "Failed to initialize SeedVR2 engine or download required weights."
            return

    # Determine assigned CUDA device and check VRAM capacity
    import torch
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not torch.cuda.is_available() or count == 0:
        err = "CUDA is not available. SeedVR2 requires an NVIDIA GPU.\n"
        yield f"❌ {err}", True, False, err
        return

    vram_gb = 0
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass

    # Auto-optimize VRAM if on 8GB-12GB GPU (e.g. RTX 5050)
    if blocks_to_swap == 0 and vram_gb > 0:
        if vram_gb <= 8.5:
            blocks_to_swap = 16
        elif vram_gb <= 12.5:
            blocks_to_swap = 8

    effective_batch_size = batch_size if is_video else 1
    if is_video and vram_gb > 0 and vram_gb <= 8.5 and effective_batch_size > 2:
        effective_batch_size = 2

    if is_video:
        if gpu_select == "Auto":
            assigned_device = ",".join(str(i) for i in range(count))
        else:
            dev = gpu_select.replace("GPU ", "").strip()
            assigned_device = dev if dev.isdigit() else "0"
    else:
        # Images use single GPU to avoid temporal slicing deadlock
        if gpu_select != "Auto":
            dev = gpu_select.replace("GPU ", "").strip()
            assigned_device = dev if dev.isdigit() else "0"
        else:
            assigned_device = "0"

    output_format = "mp4" if is_video else "png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    python_bin = get_python_exe()
    cmd = [
        python_bin,
        str(CLI_SCRIPT),
        str(Path(input_path).resolve()),
        "--output", str(Path(output_path).resolve()),
        "--output_format", output_format,
        "--resolution", str(resolution),
        "--dit_model", dit_model,
        "--cuda_device", str(assigned_device),
        "--batch_size", str(effective_batch_size),
        "--seed", "42",
        "--color_correction", color_correction,
        "--attention_mode", "sdpa",
        "--input_noise_scale", "0.0",
        "--latent_noise_scale", "0.0",
        "--video_backend", "ffmpeg" if is_video else "opencv",
        "--model_dir", str(MODEL_DIR)
    ]

    if is_video:
        if chunk_size > 0:
            cmd.extend(["--chunk_size", str(chunk_size)])
        if temporal_overlap > 0:
            cmd.extend(["--temporal_overlap", str(temporal_overlap)])

    if vae_tiled:
        cmd.extend([
            "--vae_encode_tiled",
            "--vae_decode_tiled",
            "--vae_encode_tile_size", str(tile_size),
            "--vae_decode_tile_size", str(tile_size),
            "--vae_encode_tile_overlap", "128",
            "--vae_decode_tile_overlap", "128"
        ])

    if blocks_to_swap > 0:
        cmd.extend(["--blocks_to_swap", str(blocks_to_swap), "--swap_io_components", "--dit_offload_device", "cpu"])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

    launch_msg = (
        f"\n--- Launching SeedVR2 Super-Resolution ---\n"
        f"Input: {os.path.basename(input_path)}\n"
        f"Target Resolution: {resolution}p | Device: GPU {assigned_device} | Model: {dit_model}\n"
        f"VRAM Detected: ~{int(vram_gb)}GB | BlockSwap: {blocks_to_swap} blocks | Batch: {effective_batch_size}\n"
    )
    yield launch_msg, False, False, ""

    recent_lines = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            cwd=str(SEEDVR2_DIR)
        )

        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            recent_lines.append(line)
            if len(recent_lines) > 50:
                recent_lines.pop(0)
            yield line, False, False, ""

        proc.wait()

        if proc.returncode == 0:
            if os.path.exists(output_path):
                yield "", True, True, output_path
                return

            # Check for possible _upscaled file in the same directory
            base_p = Path(output_path)
            stem = base_p.stem
            ext = base_p.suffix
            alt_path = base_p.parent / f"{stem}_upscaled{ext}"
            if alt_path.exists():
                import shutil
                shutil.move(str(alt_path), output_path)
                yield "", True, True, output_path
                return

            candidates = list(base_p.parent.glob(f"*{ext}"))
            if candidates:
                latest = max(candidates, key=os.path.getmtime)
                yield "", True, True, str(latest)
                return

            err_msg = f"SeedVR2 process finished with code 0 but expected output {output_path} was not found."
            yield f"\n❌ {err_msg}\n", True, False, err_msg
        else:
            err_details = "".join(recent_lines).strip()
            err_msg = f"SeedVR2 process failed with exit code {proc.returncode}.\n{err_details}"
            yield f"\n❌ {err_msg}\n", True, False, err_msg

    except Exception as e:
        err_msg = f"Exception during SeedVR2 execution: {str(e)}"
        yield f"\n❌ {err_msg}\n", True, False, err_msg


def run_seedvr2_upscale(
    input_path: str,
    output_path: str,
    is_video: bool,
    resolution: int = 1080,
    gpu_select: str = "Auto",
    chunk_size: int = 45,
    temporal_overlap: int = 2,
    batch_size: int = 5,
    color_correction: str = "lab",
    vae_tiled: bool = True,
    tile_size: int = 1024,
    dit_model: str = DEFAULT_DIT_MODEL,
    blocks_to_swap: int = 0,
    log_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str]:
    """
    Executes inference_cli.py with streaming real-time console logs.
    
    Returns:
        (success: bool, final_output_path_or_error: str)
    """
    last_success = False
    last_res = ""
    for line, finished, success, res in stream_seedvr2_upscale(
        input_path=input_path,
        output_path=output_path,
        is_video=is_video,
        resolution=resolution,
        gpu_select=gpu_select,
        chunk_size=chunk_size,
        temporal_overlap=temporal_overlap,
        batch_size=batch_size,
        color_correction=color_correction,
        vae_tiled=vae_tiled,
        tile_size=tile_size,
        dit_model=dit_model,
        blocks_to_swap=blocks_to_swap
    ):
        if line and log_callback:
            try:
                log_callback(line)
            except Exception:
                try:
                    clean_line = line.encode("ascii", errors="replace").decode("ascii")
                    log_callback(clean_line)
                except Exception:
                    pass
        if finished:
            last_success = success
            last_res = res
    return last_success, last_res
