from src.data_loader import load_config, load_embeddings_and_metadata
from src.build_index import build_index, log_to_mlflow


def main():
    print("=" * 50)
    print("  Pipeline F4: Build FAISS Index")
    print("=" * 50)

    params = load_config()

    print("\n[1/3] Cargando embeddings y metadatos...")
    data = load_embeddings_and_metadata(params)

    print("\n[2/3] Construyendo indice FAISS...")
    stats = build_index(params, data)

    print("\n[3/3] Registrando en MLflow...")
    log_to_mlflow(params, stats)

    print("\n" + "=" * 50)
    print("  Pipeline completado.")
    print(f"  Index: {params['faiss']['index_path']}")
    print(f"  Muestras: {stats['n_samples']}")
    print(f"  Tiempo: {stats['build_time_s']}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
