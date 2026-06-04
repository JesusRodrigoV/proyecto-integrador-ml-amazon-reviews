# ARQUITECTURA DEL REPOSITORIO Y MLOPS - PROYECTO "AMAZON REVIEWS" (PARA IA AGENTE)

## 1. PARÁMETRO GENERAL DE OPERACIÓN
Todas las rutas generadas en tus fragmentos de código deben ser relativas a la raíz de este repositorio. Este esquema no es una simple convención organizativa; es una arquitectura diseñada para separar la configuración del entorno, el seguimiento de métricas y la persistencia de datos, garantizando la reproducibilidad operativa en infraestructuras volátiles como Google Colab.

## 2. ÁRBOL DE DIRECTORIOS OBLIGATORIO Y SUS RESPONSABILIDADES

* `data/`
    * **Propósito:** Almacén estrictamente reservado para los datos crudos, submuestreos procesados y metadatos espaciales.
    * **Regla:** Aquí residen los archivos `.jsonl` originales y las particiones estandarizadas en `.csv` u `.orc`. Nunca se debe hacer *commit* de estos archivos a Git; deben gestionarse mediante `DVC`.
    * **Ejemplo de contenido:** `raw_office_products.csv`, `train_split_seed42.csv`.

* `notebooks/`
    * **Propósito:** Entornos de experimentación, Análisis Exploratorio de Datos (EDA) y prototipado visual.
    * **Regla:** El código aquí no es de producción. Sirve para interpretar clústeres no supervisados, evaluar distribuciones (inflación de ratings) y documentar decisiones matemáticas previas a la modularización.

* `src/`
    * **Propósito:** El núcleo lógico del sistema. Contiene los scripts Python `.py` modulares y reutilizables.
    * **Regla:** Dividido en módulos funcionales: preprocesamiento, vectorización TF-IDF, entrenamiento de ensamblajes, inferencia con Transformers y generación de embeddings. Ningún script aquí debe depender de variables globales o rutas absolutas quemadas en el código (*hardcoded*).

* `models/`
    * **Propósito:** Persistencia de artefactos entrenados.
    * **Regla:** Contiene modelos serializados (ej. `.pkl`, `.joblib`), pesos destilados de la red neuronal y los índices vectoriales serializados de FAISS/ChromaDB. Estos archivos pesados son rastreados exclusivamente por `DVC` o el registro local de `MLflow`.

* `reports/`
    * **Propósito:** Salidas documentales y evidencia estadística.
    * **Regla:** Destino final para las métricas exportadas, tablas comparativas (especialmente la fila obligatoria del benchmark de AutoML), matrices de confusión generadas y el documento final de análisis en formato Markdown/PDF.

* `app/` (o `demo/`)
    * **Propósito:** Prototipo mínimo viable (MVP) de interacción.
    * **Regla:** Aloja el script de ejecución de la Demo (ej. vía Streamlit o FastAPI). Este componente carga el índice vectorial desde `/models`, procesa la consulta del usuario a través de `/src` y retorna la recuperación semántica (RAG) en una interfaz controlada.

* `config/`
    * **Propósito:** Parametrización centralizada del proyecto.
    * **Regla:** Debe contener archivos YAML o JSON (`params.yaml`). Aquí se definen obligatoriamente las semillas de aleatoriedad (`random_state`), el tamaño del submuestreo de la red, los hiperparámetros de los modelos clásicos y los umbrales de corte de longitud de texto (`word_count`).

* `README.md`
    * **Propósito:** Manifiesto de ejecución. Documentación sobre cómo clonar, iniciar el entorno y reproducir el pipeline completo desde la extracción de características hasta la inferencia RAG.

* `REQUIREMENTS.txt`
    * **Propósito:** Congelación estricta de dependencias. Garantiza que el entorno Python (pandas, scikit-learn, LightGBM, transformers, faiss, mlflow, dvc, lazypredict) sea idéntico en cualquier máquina o instancia en la nube.