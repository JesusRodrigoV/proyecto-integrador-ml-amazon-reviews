import time
import pickle
import numpy as np
import faiss
import mlflow
from pathlib import Path

from src.data_loader import load_config, load_embeddings_and_metadata


SENTIMENT_MAP = {0: "Negativo", 1: "Neutro", 2: "Positivo"}


def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def build_index(params, data):
    emb = data["embeddings"]
    metadata = data["metadata"]
    n_samples = data["n_samples"]
    dim = emb.shape[1]

    start = time.time()

    emb_norm = normalize(emb)
    index = faiss.IndexFlatIP(dim)
    index.add(emb_norm.astype(np.float32))

    build_time = time.time() - start
    index_size_mb = Path(params["faiss"]["index_path"]).parent

    index_dir = Path(params["faiss"]["index_path"]).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(params["faiss"]["index_path"]))

    id_map = []
    for m in metadata:
        id_map.append({
            "review_id": m["review_id"],
            "text": m["text"],
            "sentiment_label": SENTIMENT_MAP[m["sentiment"]],
            "sentiment_code": m["sentiment"],
        })

    with open(params["faiss"]["id_map_path"], "wb") as f:
        pickle.dump(id_map, f)

    print(f"[BUILD] Index FAISS creado: {n_samples} vectores, dim={dim}")
    print(f"[BUILD] Tiempo: {build_time:.2f}s")
    print(f"[BUILD] Index: {params['faiss']['index_path']}")
    print(f"[BUILD] Id map: {params['faiss']['id_map_path']}")

    return {
        "index": index,
        "id_map": id_map,
        "build_time_s": round(build_time, 2),
        "n_samples": n_samples,
        "dim": dim,
        "index_size_mb": round(
            Path(params["faiss"]["index_path"]).stat().st_size / (1024 * 1024), 2
        ),
    }


def log_to_mlflow(params, stats):
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="build_index"):
        mlflow.log_params({
            "n_samples": stats["n_samples"],
            "dim": stats["dim"],
            "index_type": "IndexFlatIP",
            "metric": "cosine (L2 norm + inner product)",
        })
        mlflow.log_metrics({
            "build_time_s": stats["build_time_s"],
            "index_size_mb": stats["index_size_mb"],
        })
        mlflow.log_artifacts(str(Path(params["faiss"]["index_dir"])), artifact_path="faiss_index")

    print(f"[MLflow] Run logged: build_index")


def main():
    params = load_config()
    data = load_embeddings_and_metadata(params)
    stats = build_index(params, data)
    log_to_mlflow(params, stats)


if __name__ == "__main__":
    main()
