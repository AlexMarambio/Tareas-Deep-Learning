# Tareas Deep Learning
 Tareas para deep learning
## Cómo correr el notebook en local
### Estructura
```
Tarea_1/
├── notebooks/
│   ├── Tarea_1_IDS_DNN.ipynb                  # notebook base (config. rápida)
│   ├── Tarea_1_IDS_DNN_valores_altos.ipynb    # notebook con presupuestos altos (final)
│   └── local_run/
│       └── IDS_Tarea1/
│           ├── data/
│           │   ├── raw/                       # 8 CSV crudos de CIC-IDS2017/2018
│           │   └── processed/
│           │       ├── fast_dev/              # artefactos de la corrida rápida
│           │       │   └── raw_clean.parquet
│           │       └── full/                  # artefactos de la corrida completa
│           │           ├── raw_clean.parquet
│           │           ├── train.parquet / val.parquet / test.parquet
│           │           ├── train_augmented.parquet
│           │           ├── scaler.joblib / label_encoder.joblib
│           │           ├── feature_names.json
│           │           └── dataset_card.json
│           └── results/
```
### 1. Requisitos
- Python 3.11+ (probado en 3.13)
- Datos crudos del CIC-IDS2017/2018 en `notebooks/local_run/IDS_Tarea1/data/raw/*.csv`

### 2. Instalar dependencias
```bash
python -m pip install --upgrade pandas scikit-learn matplotlib torch joblib
python -m pip install jupyter nbconvert nbclient ipykernel
python -m pip install optuna
python -m pip install "flaml[automl]"
python -m pip install sdv
```

### 3. Configurar el modo de corrida
En la celda de `CONFIG` del notebook:
```python
CONFIG["fast_dev_run"] = False   # True = muestra chica, para validar rápido
```

### 4. Dónde quedan los resultados
En el mismo archivo notebook pero también se pueden ver en:
```
notebooks/local_run/IDS_Tarea1/results/parte1/
notebooks/local_run/IDS_Tarea1/results/parte2/
notebooks/local_run/IDS_Tarea1/results/parte3/
```
