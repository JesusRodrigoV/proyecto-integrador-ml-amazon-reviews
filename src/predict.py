import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

from src.data_loader import load_config


SENTIMENT_LABELS = ["Negativo", "Neutro", "Positivo"]

_tokenizer = None
_model = None
_classifier = None
_params = None


def _load_models(params):
    global _tokenizer, _model, _classifier
    if _tokenizer is not None and _model is not None and _classifier is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = params["model"]["transformer_name"]
    logreg_path = params["model"]["logreg_model_path"]

    print(f"[PREDICT] Cargando {model_name} en {device}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name).to(device).eval()

    if not Path(logreg_path).exists():
        raise FileNotFoundError(
            f"No se encontro {logreg_path}. "
            "Descarga classifier.pkl desde Google Drive."
        )
    _classifier = joblib.load(logreg_path)
    print("[PREDICT] Modelos listos.")


def init(params=None):
    global _params
    if params is None:
        _params = load_config()
    else:
        _params = params
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

    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def predict(text):
    if _classifier is None or _model is None:
        raise RuntimeError("init() debe ejecutarse antes de predict()")

    emb = encode(text)
    y_pred = _classifier.predict(emb)[0]
    y_proba = _classifier.predict_proba(emb)[0]

    confidence = float(np.max(y_proba))

    return {
        "sentiment": SENTIMENT_LABELS[int(y_pred)],
        "sentiment_code": int(y_pred),
        "confidence": round(confidence, 4),
        "probabilities": {
            label: round(float(y_proba[i]), 4)
            for i, label in enumerate(SENTIMENT_LABELS)
        },
    }


def main():
    params = load_config()
    init(params)

    print("\n=== Clasificador de Sentimiento ===")
    print("Escribe 'salir' para terminar.\n")

    while True:
        text = input("Texto: ").strip()
        if text.lower() in ("salir", "exit", "quit"):
            break
        if not text:
            continue

        result = predict(text)
        print(f"\n  Prediccion: {result['sentiment']}")
        print(f"  Confianza:  {result['confidence']:.2%}")
        print(f"  Probabilidades: {result['probabilities']}\n")


if __name__ == "__main__":
    main()
