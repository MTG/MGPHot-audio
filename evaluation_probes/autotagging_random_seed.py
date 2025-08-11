#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multilabel probe for positive/negative tag prediction from audio embeddings.

- Loads ALL embeddings and labels in memory (like your previous GenomeDataset).
- Supports positive and negative tags:
    --label_type positive  -> reads genome_index_split_positive.json (field: "positive_tags")
    --label_type negative  -> reads genome_index_split_negative.json (field: "negative_tags")
- Builds a MultiLabelBinarizer from the train split only.
- Uses a small MLP with BCEWithLogitsLoss.
- Logs AUROC (macro), MAP (macro), and writes classwise MAP to JSON.
- Avoids logging metric objects directly; computes scalars at epoch end to prevent
  "No samples to concatenate" errors from TorchMetrics.
"""

import os
import json
import warnings
from pathlib import Path
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
    ConfusionMatrix,
)
from tqdm import tqdm

# =========================================================
# Seeds and vocabularies
# =========================================================

SEEDS = [40, 41, 42, 43, 44]  # five runs

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
    "Focus on Musical Arrangements", "Focus on Form", "Focus on Riffs", "Focus on Performance",
]
POSITIVE_VOCAB = set(FEATURE_NAMES) | {"Major", "Minor"}
NEGATIVE_VOCAB = {f"No {name}" for name in POSITIVE_VOCAB}

# =========================================================
# Embedding roots (same mapping as your previous dataset)
# =========================================================

VALID_EMBEDDING_TYPES = {
    "maest": "/projects/mtg/projects/genome_embeddings/genome_maest_embeddings",
    "mert": "/projects/mtg/projects/genome_embeddings/genome_mert_embeddings",
    "omar": "/projects/mtg/projects/genome_embeddings/genome_omar-rq-multicodebook_embeddings",
    "musicfm": "/projects/mtg/projects/genome_embeddings/genome_musicfm",
    "whisper": "/projects/mtg/projects/embeddings_autotagging/genome/whisper_embeddings",
    "clap": "/projects/mtg/projects/embeddings_autotagging/genome/clap_embeddings",
    "mule": "/projects/mtg/projects/genome_embeddings/genome_mule",
    # Add more if needed
}

# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_embedding_path(base_dir: Path, youtube_id: str) -> Path:
    """Return the path <base_dir>/<first3>/<youtube_id>.npy"""
    subfolder = youtube_id[:3]
    return base_dir / subfolder / f"{youtube_id}.npy"

def load_and_normalize_embedding(embedding_type: str, npy_path: Path) -> torch.Tensor:
    """
    Load .npy and match the exact shape handling used in your old dataset.
    No extra final averaging besides the specific cases below.
    """
    emb = np.load(npy_path)
    if emb.size == 0:
        raise ValueError("empty embedding")

    emb = torch.tensor(emb, dtype=torch.float32)

    if embedding_type == "maest":
        # Must end with 2304 features, allow shapes like (1, 9, 1, 2304) or (9, 2304) or (2304,)
        if emb.ndim < 2 or emb.shape[-1] != 2304:
            raise ValueError(f"unrecognized maest shape: {tuple(emb.shape)}")
        emb = emb.squeeze()  # e.g., (1, 9, 1, 2304) → (9, 2304)
        if emb.ndim == 1 and emb.shape[0] == 2304:
            pass
        elif emb.ndim == 2 and emb.shape[1] == 2304:
            emb = emb.mean(0)  # average over time -> (2304,)
        else:
            raise ValueError(f"unrecognized maest shape after squeeze: {tuple(emb.shape)}")

    elif embedding_type == "whisper":
        # Average over time/frame if 2D+
        if emb.ndim >= 2:
            emb = emb.mean(dim=0)

    elif embedding_type == "clap":
        # Average over time/frame if 2D+
        if emb.ndim >= 2:
            emb = emb.mean(dim=0)

    elif embedding_type == "mule":
        # Squeeze last axis if it exists and is 1
        if emb.ndim >= 1:
            emb = emb.squeeze(-1)

    elif embedding_type in {"omar", "musicfm", "mert"}:
        # Keep as is
        pass

    else:
        # Generic case: if (T, F, 1) then squeeze last axis
        if emb.ndim == 3 and emb.shape[-1] == 1:
            emb = emb.squeeze(-1)

    return emb  # return as 1D vector or already processed tensor

def build_encoder_from_train(index_json: Path, label_type: str):
    """
    Build MultiLabelBinarizer from the train split only.

    label_type: "positive" or "negative"
      - Sets the field name and the vocabulary filter.
    """
    with open(index_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    field = "positive_tags" if label_type == "positive" else "negative_tags"
    vocab = POSITIVE_VOCAB if label_type == "positive" else NEGATIVE_VOCAB

    pool = []
    for _, v in meta.items():
        if v.get("split") != "train":
            continue
        tags = v.get(field, [])
        tags = [t for t in tags if t in vocab]  # filter to known vocabulary
        pool.extend(tags)

    unique_labels = sorted(set(pool)) if pool else sorted(vocab)
    encoder = MultiLabelBinarizer()
    encoder.fit([unique_labels])
    class_names = list(encoder.classes_)
    return encoder, class_names

def _auto_workers() -> int:
    """Choose a reasonable num_workers for DataLoader."""
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return max(2, min(8, cpu // 2))

# =========================================================
# Dataset: load EVERYTHING in memory (like your previous class)
# =========================================================

class TagsGenomeDataset(Dataset):
    """
    Load ALL embeddings and labels into memory.

    Expects index_json entries with:
      {
        "youtube_id": "...",
        "split": "train" | "val" | "test",
        "is_official": bool,
        "positive_tags": [...]  # if label_type == "positive"
        "negative_tags": [...]  # if label_type == "negative"
      }

    Embedding path resolution follows VALID_EMBEDDING_TYPES and subfolder rule.
    """
    def __init__(self,
                 embedding_type: str,
                 index_json: Path,
                 split: str = "train",
                 label_type: str = "positive",
                 only_official: bool = False,
                 encoder: MultiLabelBinarizer = None):
        if embedding_type not in VALID_EMBEDDING_TYPES:
            raise ValueError(
                f"Embedding type '{embedding_type}' is not valid. "
                f"Available: {list(VALID_EMBEDDING_TYPES.keys())}"
            )

        self.embedding_type = embedding_type
        self.base_dir = Path(VALID_EMBEDDING_TYPES[embedding_type])
        self.index_json = Path(index_json)
        self.split = split
        self.label_type = label_type
        self.field = "positive_tags" if label_type == "positive" else "negative_tags"
        self.encoder = encoder

        with open(self.index_json, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        entries = [
            entry for entry in all_data.values()
            if entry.get("split") == split and (entry.get("is_official") or not only_official)
        ]

        self.embeddings = []
        self.labels = []
        self.ids = []

        # Skip counters for diagnostics
        missing = 0
        empty = 0
        wrong_shape = 0
        label_missing = 0

        for entry in tqdm(entries, desc=f"Loading {split} set ({label_type})"):
            youtube_id = entry.get("youtube_id")
            tags = entry.get(self.field, [])

            if not youtube_id:
                label_missing += 1
                continue

            npy_path = get_embedding_path(self.base_dir, youtube_id)
            if not npy_path.exists():
                missing += 1
                continue

            try:
                emb = load_and_normalize_embedding(self.embedding_type, npy_path)
            except Exception as e:
                warnings.warn(f"[SKIPPED] {youtube_id}: {e}")
                wrong_shape += 1
                continue

            if self.encoder is None:
                raise ValueError("Encoder is None. Provide a fitted MultiLabelBinarizer.")
            bin_labels = self.encoder.transform([tags]).squeeze()
            label = torch.tensor(bin_labels, dtype=torch.float32)

            if emb.numel() == 0:
                empty += 1
                continue

            self.embeddings.append(emb)
            self.labels.append(label)
            self.ids.append(youtube_id)

        print(f"[INFO] Loaded {len(self.embeddings)} embeddings in memory.")
        print(f"[SKIPPED] Missing: {missing}, Empty: {empty}, Wrong shape: {wrong_shape}, Label missing: {label_missing}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.ids[idx]

# =========================================================
# LightningModule
# =========================================================

class FullyConnectedProbeMultiLabel(pl.LightningModule):
    """
    Simple MLP probe for multilabel classification with BCEWithLogitsLoss.
    Logs scalars at epoch end to avoid TorchMetrics state issues.
    Writes classwise MAP to JSON in results_dir/classwise_results.
    """
    def __init__(self, input_dim: int, num_labels: int, lr: float,
                 class_names=None, dataset_name="genome", rep_type="model", iteration=0,
                 results_dir: Path = Path("results_multilabel")):
        super().__init__()
        self.save_hyperparameters(ignore=["class_names"])
        self.num_labels = num_labels

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels),
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_metrics = nn.ModuleDict({
            "train-AUROC-macro": MultilabelAUROC(num_labels=num_labels, average="macro"),
            "train-MAP-macro": MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
        })
        self.val_metrics = nn.ModuleDict({
            "val-AUROC-macro": MultilabelAUROC(num_labels=num_labels, average="macro"),
            "val-MAP-macro": MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
        })
        self.test_metrics = nn.ModuleDict({
            "test-AUROC-macro": MultilabelAUROC(num_labels=num_labels, average="macro"),
            "test-MAP-macro": MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
            "test-MAP-classwise": MultilabelAveragePrecision(num_labels=num_labels, average=None),
        })
        self.test_confusion_matrix = ConfusionMatrix(
            task="multilabel", num_labels=num_labels, threshold=0.5
        )

        self.class_names = class_names
        self.dataset_name = dataset_name
        self.rep_type = rep_type
        self.iteration = iteration
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, _):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)

        # Update metrics (do not log metric objects)
        for metric in self.train_metrics.values():
            metric.update(probs, y.int())

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # Compute -> log scalar -> reset
        for name, metric in self.train_metrics.items():
            try:
                value = metric.compute()
            except Exception:
                value = torch.tensor(float("nan"))
            self.log(name, value, prog_bar=True)
            metric.reset()

    def validation_step(self, batch, _):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)

        for metric in self.val_metrics.values():
            metric.update(probs, y.int())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        for name, metric in self.val_metrics.items():
            try:
                value = metric.compute()
            except Exception:
                value = torch.tensor(float("nan"))
            self.log(name, value, prog_bar=True)
            metric.reset()

    def test_step(self, batch, _):
        x, y, _ = batch
        logits = self(x)
        probs = torch.sigmoid(logits)

        # Update metrics; logging is done in on_test_epoch_end
        for metric in self.test_metrics.values():
            metric.update(probs, y.int())

        preds_bin = (probs >= 0.5).int()
        self.test_confusion_matrix.update(preds_bin, y.int())

    def on_test_epoch_end(self):
        # Log macro metrics as scalars and write classwise MAP to JSON
        for name, metric in self.test_metrics.items():
            try:
                if "classwise" in name and self.class_names is not None:
                    out_dir = self.results_dir / "classwise_results"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    metric_value = metric.compute().cpu().numpy()
                    out_path = out_dir / f"{self.dataset_name}_{self.rep_type}_{name}_it_{self.iteration}.json"
                    values = {cls: float(v) for cls, v in zip(self.class_names, metric_value)}
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(values, f, indent=4)

                    # Also log the mean for convenience
                    self.log(f"{name}-mean", float(np.nanmean(metric_value)), prog_bar=True)
                else:
                    value = metric.compute()
                    self.log(name, value, prog_bar=True)
            except Exception:
                # If AUROC/MAP cannot be computed (e.g., no pos/neg), log NaN instead of crashing
                self.log(name, torch.tensor(float("nan")))
            finally:
                metric.reset()

        _ = self.test_confusion_matrix.compute()
        self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# =========================================================
# One seed
# =========================================================

def run_one_seed(
    index_json: Path,
    embedding_type: str,
    seed: int,
    label_type: str,
    batch_size: int = 64,
    lr: float = 5e-4,
    max_epochs: int = 1000,
    patience: int = 50,
    use_wandb: bool = False,
    results_dir: Path = Path("results_multilabel"),
    only_official: bool = False,
):
    set_seed(seed)
    dataset_name = Path(index_json).stem

    # Build encoder from train split
    encoder, class_names = build_encoder_from_train(index_json, label_type)

    # Build datasets (everything in memory)
    train_ds = TagsGenomeDataset(
        embedding_type=embedding_type,
        index_json=index_json,
        split="train",
        label_type=label_type,
        only_official=only_official,
        encoder=encoder,
    )
    val_ds = TagsGenomeDataset(
        embedding_type=embedding_type,
        index_json=index_json,
        split="val",
        label_type=label_type,
        only_official=only_official,
        encoder=encoder,
    )
    test_ds = TagsGenomeDataset(
        embedding_type=embedding_type,
        index_json=index_json,
        split="test",
        label_type=label_type,
        only_official=only_official,
        encoder=encoder,
    )

    if len(train_ds) == 0:
        raise RuntimeError("train split is empty.")

    # Determine input dimension from the first sample
    x0, _, _ = train_ds[0]
    input_dim = int(x0.shape[-1])

    # Model
    probe = FullyConnectedProbeMultiLabel(
        input_dim=input_dim,
        num_labels=len(class_names),
        lr=lr,
        class_names=class_names,
        dataset_name=dataset_name,
        rep_type=f"{embedding_type}_{label_type}",
        iteration=seed,
        results_dir=results_dir,
    )

    # Logger (optional)
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project=f"genome-{label_type}-tags",
            name=f"{embedding_type}_{label_type}_seed_{seed}",
            group=f"{embedding_type}_{label_type}_probe_multilabel",
        )

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=max_epochs,
        precision="16-mixed",
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, mode="min")],
        log_every_n_steps=10,
        enable_checkpointing=False,
        logger=logger,
    )

    # DataLoaders with num_workers to avoid bottlenecks
    num_workers = _auto_workers()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    trainer.fit(probe, train_loader, val_loader)
    trainer.test(probe, test_loader)

    return {"class_names": class_names, "num_labels": len(class_names), "input_dim": input_dim}

# =========================================================
# All seeds
# =========================================================

def run_all_seeds(index_json: Path, embedding_type: str, label_type: str, seeds=SEEDS, **kwargs):
    summary = {}
    for seed in seeds:
        info = run_one_seed(index_json, embedding_type, seed, label_type, **kwargs)
        summary[seed] = info
    return summary

# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multilabel probe with positive or negative tags, loading everything in memory."
    )
    parser.add_argument(
        "--label_type", type=str, choices=["positive", "negative"], default="positive",
        help="Choose which tag set to use."
    )
    parser.add_argument(
        "--positive_index", type=Path, default=Path("../genome_index_split_positive.json"),
        help="Path to JSON with positive_tags."
    )
    parser.add_argument(
        "--negative_index", type=Path, default=Path("../genome_index_split_negative.json"),
        help="Path to JSON with negative_tags."
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["omar", "musicfm", "mert", "maest", "mule", "whisper", "clap"],
        help="Embedding types to run."
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--results_dir", type=Path, default=Path("results_multilabel"))
    parser.add_argument("--only_official", action="store_true", help="Filter by is_official==True.")
    args = parser.parse_args()

    index_json = args.positive_index if args.label_type == "positive" else args.negative_index

    for model in args.models:
        if model not in VALID_EMBEDDING_TYPES:
            raise ValueError(
                f"Embedding type '{model}' is not valid. "
                f"Available: {list(VALID_EMBEDDING_TYPES.keys())}"
            )
        run_all_seeds(
            index_json=index_json,
            embedding_type=model,
            label_type=args.label_type,
            seeds=SEEDS,
            batch_size=args.batch_size,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            use_wandb=args.use_wandb,
            results_dir=args.results_dir,
            only_official=args.only_official,
        )

if __name__ == "__main__":
    main()
