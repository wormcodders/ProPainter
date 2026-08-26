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
