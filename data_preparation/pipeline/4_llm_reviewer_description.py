import json
import os
import re
import math
import requests
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

# Arguments
parser = ArgumentParser(description="Music match analysis with Ollama")
parser.add_argument("--n-tasks", type=int)
parser.add_argument("--task-id", type=int)
args = parser.parse_args()
n_tasks = args.n_tasks
task_id = args.task_id

# Variables
url = "http://localhost:11434/api/chat"
model = "qwen2.5:32b"
INPUT_PATH = "no_matches_hard.json"
OUTPUT_DIR = Path(f"outputs_ollama/qwen2.5_32b_hard")

# Prompts
REASONING_PROMPT = """
You are an expert in identifying official versions of songs on YouTube based on video titles and descriptions.

Track information:
- Artist: {artist}
- Title: {title}
- Video title: {video_title}
- Video description: {video_description}

Explain whether the artist and song title in the video title or description match the given artist and title.
Provide your reasoning clearly and step by step, but do not make a final decision yet.
"""

MATCH_DECISION_PROMPT = """
Based on your reasoning above, do the artist and song title in the video title or description match the given track?

Answer only YES or NO.
"""

EXCLUSION_PROMPT = """
Does the following content contain any of the following or similar words:
"cover", "remastered", "remix", "karaoke", "instrumental", "live", "tribute", or "version"?

Content:
- Video title: "{video_title}"
- Video description: "{video_description}"

If yes, answer NO. If none of these words appear, answer YES.

Answer only YES or NO.
"""

def safe_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name)

def extract_yes_no(text):
    text = text.strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    return None

def load_no_matches(path=INPUT_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load songs
tracks = load_no_matches()
subset_size = math.ceil(len(tracks) / n_tasks)
if task_id == n_tasks - 1:
    tracks_sub = tracks[task_id * subset_size:]
else:
    tracks_sub = tracks[task_id * subset_size:(task_id + 1) * subset_size]

# MAIN
for track in tqdm(tracks_sub):
    name = f"{track['artist']} - {track['title']}"
    filename = OUTPUT_DIR / (safe_filename(name) + ".json")
    if filename.exists():
        continue

    track_data = {
        "artist": track["artist"],
        "title": track["title"],
        "match_found": False,
        "llm_video_analysis": []
    }

    for video in track.get("youtube", []):
        video_title = video["title"]
        video_description = video.get("description", "")

        # Prompt 1: reasoning
        reasoning_prompt = REASONING_PROMPT.format(
            artist=track['artist'],
            title=track['title'],
            video_title=video_title,
            video_description=video_description
        )
        r1 = requests.post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": reasoning_prompt}],
            "stream": False,
        })
        if r1.status_code != 200:
            print("❌ Error en razonamiento:", name, video_title)
            continue
        reasoning_text = r1.json()["message"]["content"].strip()

        # Prompt 2: decision
        r2 = requests.post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": reasoning_text + "\n" + MATCH_DECISION_PROMPT}],
            "stream": False,
        })
        if r2.status_code != 200:
            print("❌ Error en decisión:", name, video_title)
            continue
        match_decision = r2.json()["message"]["content"].strip()
        decision_result = extract_yes_no(match_decision)

        # Start analysis of the data
        analysis_data = {
            "video_title": video_title,
            "video_description": video_description,
            "reasoning_prompt": reasoning_prompt,
            "reasoning": reasoning_text,
            "match_decision_prompt": MATCH_DECISION_PROMPT,
            "match_decision": match_decision,
            "exclusion_prompt": None,
            "exclusion_result": None,
            "llm_match": False
        }

        if decision_result:
            # Prompt 3: forbidden words exclusion
            exclusion_prompt = EXCLUSION_PROMPT.format(
                video_title=video_title,
                video_description=video_description
            )
            r3 = requests.post(url, json={
                "model": model,
                "messages": [{"role": "user", "content": exclusion_prompt}],
                "stream": False,
            })
            if r3.status_code != 200:
                print("❌ Error en exclusión:", name, video_title)
                continue
            exclusion_result = r3.json()["message"]["content"].strip()
            exclusion_pass = extract_yes_no(exclusion_result)
            analysis_data["exclusion_prompt"] = exclusion_prompt
            analysis_data["exclusion_result"] = exclusion_result
            analysis_data["llm_match"] = exclusion_pass

            if exclusion_pass:
                track_data["match_found"] = True

        # Save all the analysed data
        track_data["llm_video_analysis"].append(analysis_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(track_data, f, indent=2, ensure_ascii=False)
