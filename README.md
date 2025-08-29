# Extended Metadata for MGPHot (audio links and more!)

This repo is still in construction....

## Purpose of this repository

This repository enables the community to use MGPHot in further research without redistributing restricted files.
The license of the original dataset forbids redistribution of derivative files and does not provide audio.
Therefore, this repository does **not** include `gene_values` or any audio.

Instead, you will:

1. Reconstruct the three canonical indices locally.
2. Verify each index with MD5 checksums.
3. Collect the audio for each track from public sources and verify the files.

What we provide:

* Get the data in two steps: `python reconstruct.py` and `python download_audio.py`.
* `data_preparation/`: scripts to collect audio and build the indices.
* `evaluation_probes/`: code to train lightweight models for evaluation.

_Compliance note: do not upload reconstructed indices or audio to this repository or any online service. The goal is reproducible use of MGPHot while respecting the original license._


## What you reconstruct
You will obtain three JSON index files:

1. `genome_index_split.json`
   **Task:** regression on `gene_values` (continuous targets).

2. `genome_index_split_positive.json`
   **Task:** positive music autotagging (binary tags from thresholds over `gene_values`).

3. `genome_index_split_negative.json`
   **Task:** negative music autotagging (complement of the positive tags).

Each index already includes the train/validation/test split in the field `split`.
MD5 files are used to guarantee that every index is **canonical** in content and formatting.

## How to reconstruct
Run the reconstruction script. 

```bash
python reconstruct.py
```

It will:
- download the Zenodo TSV with `gene_values`,
- rebuild the base index with `gene_values`,
- generate positive and negative indices,
- compare each output with its reference MD5,
- print a short report with dashed separators.



**Outputs created (plus their `.md5` files):**
- `genome_index_split.json`
- `genome_index_split_positive.json`
- `genome_index_split_negative.json`

If an MD5 does not match, the script prints it clearly.
MD5 ensures exact byte match, including field order, indentation, and the trailing newline policy.

## Download the audio


```bash
python download_audio.py
```

We have been able to conduct the downloads from our research institution under Directive (EU) 2019/790 on Copyright in the Digital Single Market, which includes text and data mining exceptions for the purposes of scientific research (Article 3).

## Repository layout
- `data_preparation/` — Clean and reliable process to obtain YouTube links and to build the indices.
- `download_audio/` — Scripts to download and verify all audio.
- `evaluation_probes/` — Training and evaluation code for the benchmark (regression and autotagging probes).
- `reconstruct.py` — Rebuilds the three indices and verifies MD5 for each.
- `genome_positive.py` / `genome_negative.py` — Convert `gene_values` to positive and negative tags.

## Contribute

Audio download is semi-automatic. If you find a wrong or broken link, please open an issue:

Please include:
```
- Artist name: <Artist>
- Track title: <Title>
- Old YouTube URL: <https://www.youtube.com/watch?v=...>
- Old YouTube ID: <...>
- New YouTube URL: <https://www.youtube.com/watch?v=...>
- New YouTube ID: <...>
- Notes (optional): <...>
```

## Citation

If you use this repository in research, please cite the paper:

```bibtex

```

and the original dataset:

```bibtex
@article{oramas2025mgphot,
  author    = {Oramas, Sebastian and Gouyon, Fabien and Hogan, Stuart and Landau, Chris and Ehmann, Anahid},
  title     = {MGPHot: A Dataset of Musicological Annotations for Popular Music (1958--2022)},
  journal   = {Transactions of the International Society for Music Information Retrieval},
  volume    = {8},
  number    = {1},
  pages     = {108--120},
  year      = {2025},
}
```

## License

- The code in this repository is licensed under the MIT license.
- Annotation metadata from MGPHot dataset used by the code is available for [non-commerical use](URL to the dataset).
- The metadata related to mapping to YouTube is available under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
