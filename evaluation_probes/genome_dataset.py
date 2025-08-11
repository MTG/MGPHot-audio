
from torch.utils.data import DataLoader

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import pdb

class GenomeDataset(Dataset):
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

    def __init__(self, embedding_type: str, split: str = "train", only_official: bool = False):
        self.json_path = "../genome_index_split.json"

        if embedding_type not in self.VALID_EMBEDDING_TYPES:
            raise ValueError(f"Embedding type '{embedding_type}' is not valid. "
                             f"Available: {list(self.VALID_EMBEDDING_TYPES.keys())}")

        self.embedding_type = embedding_type
        self.base_dir = Path(self.VALID_EMBEDDING_TYPES[embedding_type])

        with open(self.json_path) as f:
            all_data = json.load(f)

        self.embeddings = []
        self.labels = []
        self.ids = []

        # Skip counters
        missing = 0
        empty = 0
        wrong_shape = 0
        label_mismatch = 0

        entries = [entry for entry in all_data.values()
                   if entry["split"] == split and (entry["is_official"] or not only_official)]

        for entry in tqdm(entries, desc=f"Loading {split} set"):
            youtube_id = entry["youtube_id"]
            gene_values = entry["gene_values"]
            subfolder = youtube_id[:3]
            npy_path = self.base_dir / subfolder / f"{youtube_id}.npy"

            if not npy_path.exists():
                missing += 1
                continue

            try:
                emb = np.load(npy_path)

                if emb.size == 0:
                    empty += 1
                    continue

                if self.embedding_type == "maest":
                    emb = torch.tensor(emb, dtype=torch.float32)

                    if emb.ndim < 2 or emb.shape[-1] != 2304:
                        pdb.set_trace()
                        wrong_shape += 1
                        continue

                    emb = emb.squeeze()  # e.g. (1, 9, 1, 2304) → (9, 2304)
                    if emb.ndim == 1 and emb.shape[0] == 2304:
                        pass  # already ok
                    elif emb.ndim == 2 and emb.shape[1] == 2304:
                        emb = emb.mean(0)  # average over time
                    else:
                        print(f"[SKIPPED] Unrecognized maest shape: {emb.shape}")
                        wrong_shape += 1
                        continue

                emb = torch.tensor(emb, dtype=torch.float32)

                if len(gene_values) != 58:
                    label_mismatch += 1
                    continue

                label = torch.tensor(gene_values, dtype=torch.float32)

                self.embeddings.append(emb)
                self.labels.append(label)
                self.ids.append(youtube_id)

            except Exception as e:
                print(f"[ERROR] Failed to load {npy_path}: {e}")
                continue

        print(f"[INFO] Loaded {len(self.embeddings)} preprocessed embeddings in memory.")
        print(f"[SKIPPED] Missing: {missing}, Empty: {empty}, Wrong shape: {wrong_shape}, Label mismatch: {label_mismatch}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.ids[idx]




if __name__ == "__main__":
    dataset = GenomeDataset(
            embedding_type="mule",
        split="train",
        only_official=False
    )

    print(f"[INFO] Dataset ready with {len(dataset)} samples.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for embedding, label, youtube_id in dataloader:
        if embedding.shape != torch.Size([1, 1728]):
            print(f"[DEBUG] youtube_id: {youtube_id}")
            print(f"[DEBUG] embedding shape: {embedding.shape}")
            print(f"[DEBUG] label shape: {label.shape}")

