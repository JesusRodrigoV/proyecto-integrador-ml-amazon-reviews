import pickle
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path="config/params.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_embeddings_and_metadata(params):
    embeddings_list = []
    labels_list = []
    texts_list = []

    for split in ["train", "val", "test"]:
        emb_path = Path(params["data"][f"{split}_embeddings"])
        labels_path = Path(params["data"][f"{split}_labels"])
        texts_path = Path(params["data"][f"{split}_texts"])

        if not emb_path.exists():
            raise FileNotFoundError(
                f"No se encontro {emb_path}. "
                "Descarga los embeddings desde Google Drive a models/embeddings/"
            )

        embeddings_list.append(np.load(emb_path))
        labels_list.append(np.load(labels_path))
        with open(texts_path, "rb") as f:
            texts_list.append(pickle.load(f))

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    texts = []
    for t in texts_list:
        texts.extend(t)

    n = len(texts)
    metadata = [
        {"review_id": i, "text": texts[i], "sentiment": int(labels[i])}
        for i in range(n)
    ]

    print(f"Embeddings: {embeddings.shape}")
    print(f"Labels:     {labels.shape}")
    print(f"Samples:    {n}")

    return {
        "embeddings": embeddings,
        "labels": labels,
        "texts": texts,
        "metadata": metadata,
        "n_samples": n,
    }
