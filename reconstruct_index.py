#!/usr/bin/env python3
from __future__ import annotations
import csv, json, ast, hashlib, subprocess, sys
from pathlib import Path
from collections import OrderedDict, defaultdict

# ----------------- Config -----------------
ZENODO_URL    = "https://zenodo.org/records/15372063/files/mgphot_gene_values.tsv?download="
TSV_PATH      = Path("mgphot_gene_values.tsv")

INPUT_JSON    = Path("genome_index_split_without_gene.json")   # without gene_values
OUTPUT_JSON   = Path("genome_index_split.json")                # reconstructed (with gene_values)
ORIGINAL_MD5  = Path("md5/genome_index_split_original.md5")        # expected MD5 for OUTPUT_JSON

POS_SCRIPT    = Path("genome_positive.py")
NEG_SCRIPT    = Path("genome_negative.py")

POS_OUTPUT    = Path("genome_index_split_positive.json")
NEG_OUTPUT    = Path("genome_index_split_negative.json")

POS_MD5_FILE  = Path("md5/genome_index_split_positive.md5")
NEG_MD5_FILE  = Path("md5/genome_index_split_negative.md5")

DESIRED_ORDER = [
    "artist", "title", "youtube_url", "youtube_id",
    "is_official", "gene_values", "year", "period",
    "main_artist", "split",
]
# ------------------------------------------

def print_sep():
    print("-" * 60)

def download_file(url: str, dst: Path) -> None:
    try:
        import requests  # type: ignore
        r = requests.get(url, stream=True, timeout=90)
        r.raise_for_status()
        dst.write_bytes(r.content)
    except Exception:
        import urllib.request
        with urllib.request.urlopen(url, timeout=90) as resp:
            dst.write_bytes(resp.read())

def load_tsv_gene_values(tsv_path: Path) -> dict[tuple[str, str], list[float]]:
    mapping: dict[tuple[str, str], list[float]] = {}
    dups = defaultdict(int)
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = {h.strip().lower(): h for h in (reader.fieldnames or [])}
        a_key, t_key, g_key = headers.get("artist"), headers.get("title"), headers.get("gene_values")
        if not (a_key and t_key and g_key):
            raise ValueError("TSV must have columns: artist, title, gene_values")
        for row in reader:
            artist = (row[a_key] or "").strip()
            title  = (row[t_key] or "").strip()
            raw    = (row[g_key] or "").strip()
            vals = ast.literal_eval(raw)
            vals = [float(x) for x in vals]
            mapping[(artist, title)] = vals
    return mapping

def load_json_preserve_order(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be a dict.")
    return data

def reorder_record(rec: dict) -> OrderedDict:
    out = OrderedDict()
    for k in DESIRED_ORDER:
        if k in rec:
            out[k] = rec[k]
    for k, v in rec.items():
        if k not in out:
            out[k] = v
    return out

def reconstruct(index_wo_gene: OrderedDict, gene_map: dict[tuple[str, str], list[float]]) -> OrderedDict:
    out = OrderedDict()
    for k, rec in index_wo_gene.items():
        if not isinstance(rec, dict):
            out[k] = rec
            continue
        artist = str(rec.get("artist", "")).strip()
        title  = str(rec.get("title", "")).strip()
        key = (artist, title)
        new_rec = OrderedDict(rec)
        if key in gene_map:
            new_rec["gene_values"] = gene_map[key]
        out[k] = reorder_record(new_rec)
    return out

def save_json(obj: OrderedDict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")  # trailing newline

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def read_expected_md5(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""

def toggle_trailing_newline(path: Path) -> None:
    s = path.read_text(encoding="utf-8")
    if s.endswith("\n"):
        path.write_text(s[:-1], encoding="utf-8")
    else:
        path.write_text(s + "\n", encoding="utf-8")

def try_match_md5(path: Path, expected: str) -> tuple[str, bool]:
    """Return (digest, match). Try flipping final newline once if needed."""
    d1 = md5_file(path)
    if expected and d1 == expected:
        return d1, True
    if expected:
        toggle_trailing_newline(path)
        d2 = md5_file(path)
        if d2 == expected:
            return d2, True
        # revert if still not match
        toggle_trailing_newline(path)
    return d1, False

def run_script(script: Path, args: list[str]) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    cmd = [sys.executable, str(script)] + args
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Show stdout/stderr to help debugging
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"Failed running {script}")

def main():
    # 1) TSV ready
    if not TSV_PATH.exists():
        download_file(ZENODO_URL, TSV_PATH)

    # 2) Reconstruct base index with genes
    gene_map = load_tsv_gene_values(TSV_PATH)
    index_wo = load_json_preserve_order(INPUT_JSON)
    reconstructed = reconstruct(index_wo, gene_map)
    save_json(reconstructed, OUTPUT_JSON)

    # Check MD5 for base index
    expected_base = read_expected_md5(ORIGINAL_MD5)
    d_base, m_base = try_match_md5(OUTPUT_JSON, expected_base)
    print(f"MD5 (index): {d_base}")
    print(f"Match (index): {m_base}")
    print(f"Saved (index): {OUTPUT_JSON}")
    print_sep()

    # 3) Positive
    try:
        run_script(POS_SCRIPT, ["-i", str(OUTPUT_JSON), "-o", str(POS_OUTPUT)])
        expected_pos = read_expected_md5(POS_MD5_FILE)
        d_pos, m_pos = try_match_md5(POS_OUTPUT, expected_pos)
        print(f"MD5 (positive): {d_pos}")
        print(f"Match (positive): {m_pos}")
        print(f"Saved (positive): {POS_OUTPUT}")
    except Exception as e:
        print(f"MD5 (positive): ")
        print(f"Match (positive): False")
        print(f"Saved (positive): {POS_OUTPUT}  # error: {e}")
    print_sep()

    # 4) Negative
    try:
        run_script(NEG_SCRIPT, ["-i", str(OUTPUT_JSON), "-o", str(NEG_OUTPUT)])
        expected_neg = read_expected_md5(NEG_MD5_FILE)
        d_neg, m_neg = try_match_md5(NEG_OUTPUT, expected_neg)
        print(f"MD5 (negative): {d_neg}")
        print(f"Match (negative): {m_neg}")
        print(f"Saved (negative): {NEG_OUTPUT}")
    except Exception as e:
        print(f"MD5 (negative): ")
        print(f"Match (negative): False")
        print(f"Saved (negative): {NEG_OUTPUT}  # error: {e}")
    print_sep()

if __name__ == "__main__":
    main()
