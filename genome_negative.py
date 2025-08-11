#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

FEATURE_NAMES = [
    "Vocal Register", "Vocal Timbre Thin to Full", "Vocal Breathiness", "Vocal Smoothness",
    "Vocal Grittiness", "Vocal Nasality", "Vocal Accompaniment", "Minor / Major Key Tonality",
    "Harmonic Sophistication", "Tempo", "Cut Time Feel", "Triple Meter", "Compound Meter",
    "Odd Meter", "Swing Feel", "Shuffle Feel", "Syncopation Low to High", "Backbeat",
    "Danceability", "Drum Set", "Drum Aggressiveness", "Synthetic Drums", "Percussion",
    "Electric Guitar", "Electric Guitar Distortion", "Acoustic Guitar", "String Ensemble",
    "Horn Ensemble", "Piano", "Organ", "Rhodes", "Synthesizer", "Synth Timbre", "Bass Guitar",
    "Reed Instrument", "Angry Lyrics", "Sad Lyrics", "Happy/Joyful Lyrics", "Humorous Lyrics",
    "Love/Romance Lyrics", "Social/Political Lyrics", "Abstract Lyrics", "Explicit Lyrics",
    "Live Recording", "Audio Production", "Aural Intensity", "Acoustic Sonority",
    "Electric Sonority", "Synthetic Sonority", "Focus on Lead Vocal", "Focus on Lyrics",
    "Focus on Melody", "Focus on Vocal Accompaniment", "Focus on Rhythmic Groove",
    "Focus on Musical Arrangements", "Focus on Form", "Focus on Riffs", "Focus on Performance"
]

MAJOR_MINOR_IDX = FEATURE_NAMES.index("Minor / Major Key Tonality")

def values_to_negative_tags(values, negative_thr=0.2):
    tags = []
    for i, v in enumerate(values):
        # Special handling for tonality
        if i == MAJOR_MINOR_IDX:
            # Not Major if value <= 0.8
            if v <= 0.2:
                tags.append("No Major")
            # Not Minor if value >= 0.2
            if v >= 0.8:
                tags.append("No Minor")
            continue
        # General rule: value < 0.2
        if v < negative_thr:
            tags.append(f"No {FEATURE_NAMES[i]}")
    return tags

def convert_file(in_path: Path, out_path: Path, negative_thr=0.2):
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))

    for k, item in data.items():
        values = item.get("gene_values")
        if isinstance(values, list):
            if len(values) != len(FEATURE_NAMES):
                print(f"Warning: entry {k} has {len(values)} values, "
                      f"but feature list has {len(FEATURE_NAMES)} names.")
            item["negative_tags"] = values_to_negative_tags(
                values, negative_thr=negative_thr
            )
            # Remove gene_values if you want a clean output
            item.pop("gene_values", None)
        else:
            item.setdefault("negative_tags", [])

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Create negative tags from gene_values.")
    parser.add_argument("-i", "--input", type=Path, default=Path("genome_index_split.json"),
                        help="Input JSON path.")
    parser.add_argument("-o", "--output", type=Path, default=Path("genome_index_split_neg_tags.json"),
                        help="Output JSON path.")
    parser.add_argument("--negative_thr", type=float, default=0.2,
                        help="Threshold for general negative tags (value < negative_thr).")
    parser.add_argument("--major_thr", type=float, default=0.8,
                        help="Threshold for No Major (value <= major_thr).")
    parser.add_argument("--minor_thr", type=float, default=0.2,
                        help="Threshold for No Minor (value >= minor_thr).")
    args = parser.parse_args()

    convert_file(args.input, args.output,
                 negative_thr=args.negative_thr)

if __name__ == "__main__":
    main()
