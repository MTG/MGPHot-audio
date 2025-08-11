import json
import re
from pathlib import Path

def safe_filename(name):
    return re.sub(r"[^\w\-_\. ]", "_", name)

# PATHS
INPUT_JSON = "dataset_with_youtube.json"
MATCHES_JSON = "matches.json"
DIR_FINAL = Path("qwen2.5_32b_final")
DIR_HARD = Path("qwen2.5_32b_hard")

# Load Youtube dataset
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    youtube_tracks = json.load(f)

# load matches.json
with open(MATCHES_JSON, "r", encoding="utf-8") as f:
    matches_data = json.load(f)

# Dictionary -> True
matches_dict = {
    (m["artist"].strip(), m["title"].strip()): bool(m.get("youtube"))
    for m in matches_data
}

match_yes = 0
match_no = 0
missing = 0
no_matches_list = []


for track in youtube_tracks:
    artist = track["artist"].strip()
    title = track["title"].strip()
    key = (artist, title)
    filename = safe_filename(f"{artist} - {title}") + ".json"

    path_hard = DIR_HARD / filename
    path_final = DIR_FINAL / filename

    source = None
    match_found = None

    if path_hard.exists():
        with open(path_hard, "r", encoding="utf-8") as f:
            result = json.load(f)
        match_found = result.get("match_found", False)
        source = "hard"

    elif path_final.exists():
        with open(path_final, "r", encoding="utf-8") as f:
            result = json.load(f)
        match_found = result.get("match_found", False)
        source = "final"

    elif key in matches_dict:
        match_found = matches_dict[key]
        source = "matches"

    else:
        print(f"📂 Missing: {artist} - {title}")
        missing += 1
        continue

    if match_found:
        print(f"✅ Match YES: {artist} - {title} (source: {source})")
        match_yes += 1
    else:
        print(f"❌ Match NO: {artist} - {title} (source: {source})")
        match_no += 1
        track["source"] = source
        no_matches_list.append(track)

# Save JSON
with open("manual_annotation.json", "w", encoding="utf-8") as f:
    json.dump(no_matches_list, f, indent=2, ensure_ascii=False)

# Summary
print("\n📊 Summary:")
print(f"🎬 Total with YouTube: {len(youtube_tracks)}")
print(f"✅ Matches: {match_yes}")
print(f"❌ No matches: {match_no}")
print(f"📂 Lacking.....: {missing}")
print(f"📝 Saving no_matches_youtube.json with {len(no_matches_list)} examples.")
