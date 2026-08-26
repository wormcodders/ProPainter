import os
import subprocess
import urllib.request
import sys
import shutil

def run_cmd(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def setup():
    # 1. Clone repo if needed
    if not os.path.exists("inference_propainter.py"):
        print("Cloning ProPainter...")
        run_cmd("git clone https://github.com/sczhou/ProPainter.git temp_repo")
        # Move files up
        for item in os.listdir("temp_repo"):
            s = os.path.join("temp_repo", item)
            d = os.path.join(".", item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        shutil.rmtree("temp_repo")

    # 2. Setup VENV
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        run_cmd(f"{sys.executable} -m venv venv")

    # 3. Install dependencies
    print("Installing dependencies...")
    pip_exe = os.path.join("venv", "Scripts", "pip.exe") if os.name == 'nt' else os.path.join("venv", "bin", "pip")
    run_cmd(f"{pip_exe} install -r requirements.txt")
    run_cmd(f"{pip_exe} install gradio")

    # 4. Download models
    print("Downloading models...")
    os.makedirs("weights", exist_ok=True)
    urls = {
        "ProPainter.pth": "https://huggingface.co/camenduru/ProPainter/resolve/main/ProPainter.pth",
        "recurrent_flow_completion.pth": "https://huggingface.co/camenduru/ProPainter/resolve/main/recurrent_flow_completion.pth",
        "raft-things.pth": "https://huggingface.co/camenduru/ProPainter/resolve/main/raft-things.pth",
        "i3d_rgb_imagenet.pt": "https://huggingface.co/camenduru/ProPainter/resolve/main/i3d_rgb_imagenet.pt"
    }
    
    for name, url in urls.items():
        path = os.path.join("weights", name)
        if not os.path.exists(path):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, path)
            print(f"Finished downloading {name}.")

    print("Setup complete! Run app.py using your venv to start.")

if __name__ == "__main__":
    setup()
