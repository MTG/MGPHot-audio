
import json
import random
from tqdm import tqdm

from oguz_tools import (
    clean_video_title,
    prepare_track_for_matching,
    create_title_artist_combinations_regex,
)

def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def analyze_matches(dataset, num_examples=10):
    match_tracks = []
    no_match_tracks = []

    for track in tqdm(dataset):
        found_match = False

        videos = track.get("youtube", [])
        t_title, t_artists, t_feat_artists = prepare_track_for_matching(track)
        regexes = create_title_artist_combinations_regex(t_title, t_artists, t_feat_artists)

        for video in videos:
            v_title = clean_video_title(video.get("title", "").lower())
            if any(r.search(v_title) for r in regexes):
                found_match = True
                break

        if found_match:
            match_tracks.append(track)
        else:
            no_match_tracks.append(track)

    save_json(match_tracks, "matches.json")
    save_json(no_match_tracks, "no_matches.json")

    # Summary statistics
    print("\n========== RESULTADOS ==========")
    print(f"Songs with at least one match match: {len(match_tracks)}")
    print(f"Songs without any match: {len(no_match_tracks)}")
    print("=================================\n")

    print(f"🎯 Examples of {min(num_examples, len(no_match_tracks))} songs without any match:\n")
    for example in random.sample(no_match_tracks, min(num_examples, len(no_match_tracks))):
        print("🎵", example["artist"], "-", example["title"])
        print("   🔎 Vídeos found:")
        for vid in example["youtube"]:
            print("   ·", vid["title"])
        print()

if __name__ == "__main__":
    dataset = load_dataset("dataset_with_youtube.json")
    analyze_matches(dataset)
