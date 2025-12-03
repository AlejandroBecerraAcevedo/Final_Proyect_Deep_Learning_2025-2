# Proyecto Final - Deep Learning 2025-2

**Descripción:**

Este repositorio contiene el informe final del proyecto y cuatro notebooks (Colab/Google Colab) con las arquitecturas y experimentos realizados para la clasificación/experimentación de modelos de visión (ResNet18 y Vision Transformer).

**Contenido del repositorio:**

- `Informe_Final.pdf` (o `Informe_Final.md` según el formato usado): documento del informe final del proyecto.
- `01 - exploración y preprocesado de datos.ipynb`: Notebook para exploración y preprocesado de los datos.
- `02 - Architecture_base_resNet18_default_pytorch.ipynb`: Notebook con la arquitectura base ResNet18 (implementación por defecto en PyTorch).
- `03 - Architecture_resnet18-pytorch-built.ipynb`: Notebook con la versión construida/ajustada de ResNet18 (implementación personalizada y ajustes).
- `04 - Architecture_Vision-Transformer.ipynb`: Notebook con la arquitectura Vision Transformer y experimentos asociados.

**Advertencia importante (GPU requerida para entrenamiento):**

- Se recomienda encarecidamente ejecutar los notebooks en una GPU (tarjeta gráfica) para entrenamiento y ajuste fino. Sin GPU, el entrenamiento puede tardar extremadamente mucho o no finalizar de forma práctica.
- Si dispone de GPU local, asegúrese de tener los drivers y CUDA/cuDNN compatibles instalados.
- Recomendación: ejecutar los notebooks en Google Colab o Kaggle, que ofrecen GPUs gratuitas o aceleradas (Colab Pro/Pro+ y recursos de Kaggle ofrecen mejores tiempos de ejecución).

**¿Por qué usar Colab o Kaggle?**

- Disponibilidad de GPU/TPU para acelerar el entrenamiento.
- Configuración rápida sin necesidad de instalar dependencias localmente.
- Ejecución reproducible en entornos con recursos limitados en la máquina local.

**Notas sobre los notebooks:**

- Cada notebook incluye las celdas necesarias para obtener y preparar los datos (descarga, montaje de drive si aplica, descompresión, y preprocesado). Por lo tanto, no es necesario preparar manualmente los datasets fuera del notebook.
- Orden sugerido de ejecución:
  1. `01 - exploración y preprocesado de datos.ipynb` — inspección, limpieza y creación de los conjuntos de entrenamiento/validación/prueba.
  2. `02 - Architecture_base_resNet18_default_pytorch.ipynb` — prueba inicial con ResNet18 base.
  3. `03 - Architecture_resnet18-pytorch-built.ipynb` — experimentos con la versión modificada/optimizada de ResNet18.
  4. `04 - Architecture_Vision-Transformer.ipynb` — experimentos con Vision Transformer.
- Cada notebook está diseñado para ser ejecutado de principio a fin; algunas celdas descargan datos automáticamente o piden montar Google Drive (en Colab) para acceder a datasets grandes.

**Dependencias y entorno**

- Recomendado: usar un entorno con Python 3.8+ y bibliotecas comunes de Deep Learning como `torch`, `torchvision`, `tqdm`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn` y `timm` (si se usa). En Colab muchas de estas ya están instaladas.
- Si ejecuta localmente, cree y active un entorno virtual, luego instale las dependencias con `pip install -r requirements.txt` (si se proporciona) o instale manualmente las librerías necesarias.

**Resultados y reproducibilidad**

- Los notebooks guardan checkpoints del modelo y métricas en carpetas locales o en Google Drive si se monta. Revise las celdas iniciales de configuración en cada notebook para ajustar rutas y opciones de guardado.

**Video explicativo:**

- Link al video explicativo (poner aquí el enlace):

  [INCLUIR AQUÍ EL LINK DEL VIDEO EXPLICATIVO]

**Créditos y autoría**

- Autor: Alejandro Becerra Acevedo
- Curso: Deep Learning — Semestre 2025-2
