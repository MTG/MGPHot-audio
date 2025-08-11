import os
import re
import json
import time
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

# -------------- Config ----------------

INPUT_JSON = "genome_index_split.json"
ROOT_DIR = "/mnt/md1/genome_audio"
COOKIES_PATH = "cookies.txt"  # Ruta al archivo cookies.txt

# -------------- Helpers ----------------

def escape_ansi(text):
    return re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", text)

def get_subfolder_from_ytid(yt_id):
    return yt_id[:3]

def get_paths(yt_id, root_dir):
    subfolder = get_subfolder_from_ytid(yt_id)
    base_path = os.path.join(root_dir, subfolder)
    os.makedirs(base_path, exist_ok=True)
    audio_path = os.path.join(base_path, f"{yt_id}.mp3")
    meta_path = os.path.join(base_path, f"{yt_id}.meta")
    log_path = os.path.join(base_path, f"{yt_id}.log")
    return audio_path, meta_path, log_path

# -------------- Download ----------------

def download_audio_mp3(yt_id, root_dir, force_failed=False):
    url = f"https://www.youtube.com/watch?v={yt_id}"
    audio_path, meta_path, log_path = get_paths(yt_id, root_dir)

    if os.path.exists(audio_path) and os.path.exists(meta_path):
        return "file exists"

    if os.path.exists(log_path) and not force_failed:
        return "download previously failed"

    base = audio_path.rsplit(".", 1)[0]
    for ext in [".webm", ".m4a", ".mp4", ".part", ".temp"]:
        temp_file = base + ext
        if os.path.exists(temp_file):
            os.remove(temp_file)

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": audio_path.replace(".mp3", ".%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "source_address": "0.0.0.0",
        "force_ipv4": True,
        "quiet": False,
        "no_warnings": True,
        "cookiefile": COOKIES_PATH,
        "progress_hooks": [lambda d: print(f"  ▶️ yt-dlp status: {d['status']}") if d["status"] != "downloading" else None],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            info = ydl.extract_info(url, download=False)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 10000:
            return "downloaded"
        else:
            return "failed: no .mp3 created"

    except (DownloadError, Exception) as e:
        with open(log_path, "w") as f:
            f.write(escape_ansi(str(e)))
        return f"error: {str(e)}"

# -------------- Main ----------------

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    tracks = list(json.load(f).values())

total = len(tracks)
success = 0
failed = 0
failed_links = []
t0 = time.time()

for i, track in enumerate(tracks, 1):
    url = track.get("youtube_url", "")
    match = re.search(r"(?:v=|youtu\.be/)([\w\-]{11})", url)
    if not match or len(match.group(1)) != 11:
        failed += 1
        failed_links.append((url, "malformed URL"))
        print(f"[{i}/{total}] ❌ URL malformada: {url}")
        continue

    yt_id = match.group(1)
    status = download_audio_mp3(yt_id, ROOT_DIR)

    if status in ["downloaded", "file exists"]:
        success += 1
    else:
        failed += 1
        failed_links.append((url, status))

    print(f"[{i}/{total}] 🎵 {yt_id} - {status} | ✅ {success} | ❌ {failed}")

# Mostrar enlaces fallidos por pantalla
if failed_links:
    print("\n🚫 Enlaces fallidos:")
    for url, reason in failed_links:
        print(f" - {url}  # {reason}")

print("\n🎉 Finalizado")
print(f"✅ Descargados: {success}")
print(f"❌ Fallidos: {failed}")
print(f"⏱️ Tiempo total: {round(time.time() - t0, 1)} s")
