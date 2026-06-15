import pickle
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

from src.data_loader import load_config
from src.build_index import normalize


SENTIMENT_MAP = {0: "Negativo", 1: "Neutro", 2: "Positivo"}

_tokenizer = None
_model = None
_index = None
_id_map = None
_params = None


def _load_models(params):
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = params["model"]["transformer_name"]

    print(f"[SEARCH] Cargando {model_name} en {device}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name).to(device).eval()
    print("[SEARCH] Modelo listo.")


def _load_index(params):
    global _index, _id_map
    if _index is not None and _id_map is not None:
        return

    index_path = params["faiss"]["index_path"]
    id_map_path = params["faiss"]["id_map_path"]

    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"No se encontro {index_path}. Ejecuta build_index.py primero."
        )
    if not Path(id_map_path).exists():
        raise FileNotFoundError(
            f"No se encontro {id_map_path}. Ejecuta build_index.py primero."
        )

    _index = faiss.read_index(str(index_path))
    with open(id_map_path, "rb") as f:
        _id_map = pickle.load(f)

    print(f"[SEARCH] Index cargado: {_index.ntotal} vectores.")


def init(params=None):
    global _params
    if params is None:
        _params = load_config()
    else:
        _params = params
    _load_index(_params)
    _load_models(_params)


def encode(text):
    global _tokenizer, _model
    device = next(_model.parameters()).device

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = _model(**inputs)

    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return normalize(emb)


def search(query, top_k=10):
    if _index is None or _id_map is None or _model is None:
        raise RuntimeError("init() debe ejecutarse antes de search()")

    query_emb = encode(query)
    scores, indices = _index.search(query_emb.astype(np.float32), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        meta = _id_map[idx]
        results.append({
            "rank": i + 1,
            "score": float(scores[0][i]),
            "text": meta["text"],
            "sentiment": meta["sentiment_label"],
            "review_id": meta["review_id"],
        })

    return results


def main():
    params = load_config()
    init(params)

    print("\n=== Busqueda Semantica ===")
    print("Escribe 'salir' para terminar.\n")

    while True:
        query = input("Consulta: ").strip()
        if query.lower() in ("salir", "exit", "quit"):
            break
        if not query:
            continue

        results = search(query, top_k=params["search"]["default_top_k"])
        print(f"\n  Top-{len(results)} resultados:\n")
        for r in results:
            print(f"  #{r['rank']} | score: {r['score']:.4f} | {r['sentiment']}")
            print(f"    {r['text'][:120]}...\n" if len(r['text']) > 120 else f"    {r['text']}\n")


if __name__ == "__main__":
    main()
