#!/usr/bin/env python3
"""
Split genome_index.json into train/val/test with balanced label distribution.
Labels include
  * 11 gene value bins (0.0 .. 1.0, step 0.1)
  * Official‐song flag
  * Five‑year period of the song year

The year is read from dataset_with_youtube.json, which must contain
objects like {"artist": "...", "title": "...", "year": 2007}.

The script looks for a split where the maximum absolute difference
between the global frequency of any label and the frequency inside each
subset is below THRESH (0.018).

Output: genome_index_split.json with a new "split" field on each song.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from oguz_tools import soft_clean_text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GENOME_PATH = Path("genome_index.json")
YOUTUBE_PATH = Path("dataset_with_youtube.json")
OUT_PATH = Path("genome_index_split.json")

SPAN_YEARS = 1       # length of one period bucket
TEST_FRACTION = 0.40 # first split: train vs temp
VAL_FRACTION = 0.50  # second split: temp -> val vs test
THRESH = 0.02      # maximum tolerated gap per label
MAX_ATTEMPTS = 2000   # maximum seeds to try

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_main_artist_name(artist_raw: str) -> str:
    """Return the main artist name, cleaned, without featuring tags."""
    main_part = re.split(r"(?i)\s*(feat\.?|ft\.?|featuring|&|with)\s+", artist_raw)[0]
    return soft_clean_text(main_part)


def year_to_period(year: Union[int, None], span: int = SPAN_YEARS) -> str:
    """Map a year to the first year of its period span. Unknown stays 'unknown'."""
    if year is None:
        return "unknown"
    return str((year // span) * span)  # e.g. 1997 -> "1995"


def load_year_mapping(path: Path) -> Dict[Tuple[str, str], int]:
    """Build (artist,title) -> year mapping from the YouTube dataset."""
    mapping: Dict[Tuple[str, str], int] = {}
    if not path.exists():
        print(f"⚠️  Warning: {path} not found. All songs will have unknown year.")
        return mapping

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        artist_c = soft_clean_text(rec.get("artist", ""))
        title_c = soft_clean_text(rec.get("title", ""))
        try:
            year_val = int(rec["year"])
        except (KeyError, ValueError, TypeError):
            continue
        mapping[(artist_c, title_c)] = year_val
    return mapping


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("📦 Loading genome index …")
with GENOME_PATH.open("r", encoding="utf-8") as f:
    songs: Dict[str, dict] = json.load(f)

print("📦 Loading YouTube year data …")
year_map = load_year_mapping(YOUTUBE_PATH)

# Add year and period info to each song
period_set = set()
for sid, song in songs.items():
    artist_c = soft_clean_text(song["artist"])
    title_c = soft_clean_text(song["title"])
    year_val = year_map.get((artist_c, title_c))
    song["year"] = year_val  # may be None
    period = year_to_period(year_val)
    song["period"] = period
    period_set.add(period)

periods: List[str] = sorted(period_set)
period_to_idx = {p: i for i, p in enumerate(periods)}

print(f"🕑 Period buckets: {periods}")

# ---------------------------------------------------------------------------
# Build artist → song ids index
# ---------------------------------------------------------------------------
artist_to_ids: Dict[str, List[str]] = defaultdict(list)
for sid, song in songs.items():
    main_artist = extract_main_artist_name(song["artist"])
    song["main_artist"] = main_artist
    artist_to_ids[main_artist].append(sid)

# ---------------------------------------------------------------------------
# Prepare multilabel vectors per artist
# ---------------------------------------------------------------------------
levels = [i / 10 for i in range(11)]  # 0.0 … 1.0

artist_labels: Dict[str, np.ndarray] = {}
for artist, sids in artist_to_ids.items():
    # Gene bins and official flag per song
    X = np.array([songs[sid]["gene_values"] for sid in sids], dtype=np.float32)
    Y_genes = np.concatenate([np.isclose(X, v) for v in levels], axis=1)
    Y_flag = np.array([[1 if songs[sid]["is_official"] else 0] for sid in sids])

    # Period one‑hot per song
    Y_periods = np.zeros((len(sids), len(periods)), dtype=int)
    for i, sid in enumerate(sids):
        idx = period_to_idx[songs[sid]["period"]]
        Y_periods[i, idx] = 1

    Y = np.hstack([Y_genes.astype(int), Y_flag, Y_periods])
    artist_labels[artist] = Y.mean(axis=0)  # average per artist

# ---------------------------------------------------------------------------
# Try to find a good split
# ---------------------------------------------------------------------------
artists = list(artist_labels.keys())
X_artists = np.array([artist_labels[a] for a in artists])

success = False
for attempt in range(1, MAX_ATTEMPTS + 1):
    random_state = 42 + attempt
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION, random_state=random_state
    )
    train_idx, temp_idx = next(splitter.split(X_artists, X_artists))

    artists_train = [artists[i] for i in train_idx]
    artists_temp = [artists[i] for i in temp_idx]

    splitter2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION, random_state=random_state
    )
    val_idx, test_idx = next(
        splitter2.split(
            [X_artists[i] for i in temp_idx], [X_artists[i] for i in temp_idx]
        )
    )

    artists_val = [artists_temp[i] for i in val_idx]
    artists_test = [artists_temp[i] for i in test_idx]

    # Build final song‑level split map
    final_split: Dict[str, str] = {}
    for a in artists_train:
        for sid in artist_to_ids[a]:
            final_split[sid] = "train"
    for a in artists_val:
        for sid in artist_to_ids[a]:
            final_split[sid] = "val"
    for a in artists_test:
        for sid in artist_to_ids[a]:
            final_split[sid] = "test"

    # Evaluate label distribution on song level
    all_ids = list(final_split.keys())
    X_all = np.array([songs[sid]["gene_values"] for sid in all_ids], dtype=np.float32)
    Y_genes_all = np.concatenate([np.isclose(X_all, v) for v in levels], axis=1)
    Y_flag_all = np.array([[1 if songs[sid]["is_official"] else 0] for sid in all_ids])

    Y_periods_all = np.zeros((len(all_ids), len(periods)), dtype=int)
    for i, sid in enumerate(all_ids):
        idx = period_to_idx[songs[sid]["period"]]
        Y_periods_all[i, idx] = 1

    Y_all = np.hstack([Y_genes_all.astype(int), Y_flag_all, Y_periods_all])

    sid_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    train_idx_full = [sid_to_idx[sid] for sid in all_ids if final_split[sid] == "train"]
    val_idx_full = [sid_to_idx[sid] for sid in all_ids if final_split[sid] == "val"]
    test_idx_full = [sid_to_idx[sid] for sid in all_ids if final_split[sid] == "test"]

    def label_freq(indices: List[int]) -> np.ndarray:
        return Y_all[indices].mean(axis=0)

    overall = label_freq(list(range(len(all_ids))))
    g_train = np.abs(label_freq(train_idx_full) - overall).max()
    g_val = np.abs(label_freq(val_idx_full) - overall).max()
    g_test = np.abs(label_freq(test_idx_full) - overall).max()

    print(
        f"Attempt {attempt:3d}/{MAX_ATTEMPTS}  gap_train={g_train:.4f}  "
        f"gap_val={g_val:.4f}  gap_test={g_test:.4f}"
    )

    if g_train < THRESH and g_val < THRESH and g_test < THRESH:
        success = True
        break

if not success:
    raise RuntimeError(
        f"❌ Could not reach gap < {THRESH} after {MAX_ATTEMPTS} attempts."
    )

# ---------------------------------------------------------------------------
# Save result
# ---------------------------------------------------------------------------
for sid in songs:
    songs[sid]["split"] = final_split.get(sid, "ignore")

with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(songs, f, indent=2, ensure_ascii=False)

print("✅ Saved to", OUT_PATH)
print("Max absolute difference vs overall:")
print("  train:", round(g_train, 4))
print("  val  :", round(g_val, 4))
print("  test :", round(g_test, 4))
print("✅ Distribution check passed.")
