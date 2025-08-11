import json
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from collections import defaultdict
import random

from genome_dataset import GenomeDataset

# Semillas fijas
SEEDS = [40, 41, 42, 43, 44, 45]

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_cat(pairs):
    """Concatena listas de tensores si no están vacías. Devuelve (y_hat, y)."""
    filt = [(a, b) for a, b in pairs if a.numel() > 0 and b.numel() > 0]
    if not filt:
        return None, None
    a_list, b_list = zip(*filt)
    return torch.cat(a_list), torch.cat(b_list)


def compute_qwk_per_feature(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Calcula QWK por característica.
    - y_true, y_pred: tensores en CPU con forma (N, num_features)
    - Se discretiza a 11 valores {0, 1, ..., 10} equivalentes a {0.0, 0.1, ..., 1.0}
    """
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    qwks = []
    for i in range(y_true_np.shape[1]):
        t = np.clip(np.round(y_true_np[:, i] * 10), 0, 10).astype(int)
        p = np.clip(np.round(y_pred_np[:, i] * 10), 0, 10).astype(int)
        qwk = cohen_kappa_score(t, p, weights="quadratic")
        qwks.append(float(qwk))
    return qwks


class SimpleRegressionProbe(pl.LightningModule):
    def __init__(self, input_dim: int, label_dim: int, hidden_dim: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, label_dim)
        )
        self.criterion = nn.MSELoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_metrics_dict = {}

    def forward(self, x):
        return self.probe(x)

    def training_step(self, batch, _):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y, _ = batch
        y_hat = self(x)
        self.validation_step_outputs.append((y_hat.detach().cpu(), y.detach().cpu()))
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        y_hat, y = safe_cat(self.validation_step_outputs)
        if y_hat is None:
            return
        # Aquí puedes calcular métricas de validación adicionales si quieres
        self.validation_step_outputs.clear()

    def test_step(self, batch, _):
        x, y, _ = batch
        y_hat = self(x)
        self.test_step_outputs.append((y_hat.detach().cpu(), y.detach().cpu()))

    def on_test_epoch_end(self):
        y_hat, y = safe_cat(self.test_step_outputs)
        if y_hat is None:
            return

        # MSE y RMSE por característica
        mse_per_feature = ((y - y_hat) ** 2).mean(dim=0).tolist()
        rmse_per_feature = [val ** 0.5 for val in mse_per_feature]

        # QWK por característica
        qwk_per_feature = compute_qwk_per_feature(y, y_hat)

        # Construir el diccionario por característica
        self.test_metrics_dict = {}
        for i, name in enumerate(FEATURE_NAMES):
            key = f"{i:02d}_{name}"
            self.test_metrics_dict[key] = {
                "MSE": mse_per_feature[i],
                "RMSE": rmse_per_feature[i],
                "QWK": qwk_per_feature[i]
            }

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def run_one_seed(model_name: str, seed: int, lr: float = 5e-4, hidden_dim: int = 512,
                 batch_size: int = 32, label_dim: int = 58):
    set_seed(seed)

    train_ds = GenomeDataset(model_name, split="train", only_official=False)
    val_ds   = GenomeDataset(model_name, split="val",   only_official=False)
    test_ds  = GenomeDataset(model_name, split="test",  only_official=False)

    input_dim = train_ds[0][0].shape[-1]
    probe = SimpleRegressionProbe(input_dim, label_dim, hidden_dim, lr)
    probe.hparams.model = model_name

    logger = WandbLogger(
        project="genome-regression",
        entity="mtg-upf",
        name=f"{model_name}_seed_{seed}",
        group=f"{model_name}_probe"
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=1000,
        precision="16-mixed",
        logger=logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=50, mode="min")],
        log_every_n_steps=10,
        enable_checkpointing=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    trainer.fit(probe, train_loader, val_loader)
    trainer.test(probe, test_loader)

    return probe.test_metrics_dict


def run_all_seeds(model_name: str):
    # Estructura: {feature_key: {"MSE": [], "RMSE": [], "QWK": []}}
    feature_lists = defaultdict(lambda: {"MSE": [], "RMSE": [], "QWK": []})

    for seed in SEEDS:
        metrics_dict = run_one_seed(model_name, seed)
        for feat_key, metric_vals in metrics_dict.items():
            feature_lists[feat_key]["MSE"].append(metric_vals["MSE"])
            feature_lists[feat_key]["RMSE"].append(metric_vals["RMSE"])
            feature_lists[feat_key]["QWK"].append(metric_vals["QWK"])

    results = {
        "seeds": SEEDS,
        "features": feature_lists
    }

    out_file = f"results_{model_name}_seeds40-45.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{out_file} escrito.")


if __name__ == "__main__":
    for model in ["mule", "omar", "musicfm", "mert", "maest"]:  #  ["clap", "whisper"]:
        run_all_seeds(model)
