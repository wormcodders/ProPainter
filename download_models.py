import os
import urllib.request

urls = {
    "ProPainter.pth": "https://huggingface.co/camenduru/ProPainter/resolve/main/ProPainter.pth",
    "recurrent_flow_completion.pth": "https://huggingface.co/camenduru/ProPainter/resolve/main/recurrent_flow_completion.pth",
    "raft-things.pth": "https://huggingface.co/camenduru/ProPainter/resolve/main/raft-things.pth"
}

os.makedirs("weights", exist_ok=True)

for name, url in urls.items():
    path = os.path.join("weights", name)
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"Finished downloading {name}.")
    else:
        print(f"{name} already exists.")

# SeedVR2 Models
try:
    from seedvr2_runner import ensure_seedvr2_models
    print("\nChecking SeedVR2 models...")
    ensure_seedvr2_models()
except Exception as e:
    print(f"Note: SeedVR2 model setup skipped or failed: {e}")
