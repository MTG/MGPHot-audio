#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from collections import Counter

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

def value_to_tag(i, v):
    """Map a value to the correct tag string."""
    if i == MAJOR_MINOR_IDX:
        if v < 0.33:
            return "Minor"
        elif v < 0.66:
            return "Ambiguous Key Mode"
        else:
            return "Major"
    else:
        if v < 0.33:
            return FEATURE_NAMES[i] + " Low"
        elif v < 0.66:
            return FEATURE_NAMES[i] + " Moderate"
        else:
            return FEATURE_NAMES[i] + " High"

def values_to_tags(values):
    return [value_to_tag(i, v) for i, v in enumerate(values)]

def convert_file(in_path: Path, out_path: Path):
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    counter = Counter()

    for k, item in data.items():
        values = item.get("gene_values")
        if isinstance(values, list):
            if len(values) != len(FEATURE_NAMES):
                print(f"Warning: entry {k} has {len(values)} values, "
                      f"but feature list has {len(FEATURE_NAMES)} names.")
            tags = values_to_tags(values)
            item["positive_tags"] = tags
            counter.update(tags)
            item.pop("gene_values", None)
        else:
            item.setdefault("positive_tags", [])

    # create global tag index sorted by count
    sorted_tags = counter.most_common()
    print(sorted_tags)

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert gene_values to positive_tags with 3-level bins.")
    parser.add_argument("-i", "--input", type=Path, default=Path("genome_index_split.json"),
                        help="Input JSON path.")
    parser.add_argument("-o", "--output", type=Path, default=Path("genome_index_split_tags.json"),
                        help="Output JSON path.")
    args = parser.parse_args()

    convert_file(args.input, args.output)

if __name__ == "__main__":
    main()
