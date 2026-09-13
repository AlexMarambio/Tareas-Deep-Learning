# Hallazgos y errores — corrida completa (`fast_dev_run = False`)

Registro de la ejecución de punta a punta de `notebooks/Tarea_1_IDS_DNN.ipynb` sobre el dataset
completo CIC-IDS2018 (8 CSV, 2.830.743 filas, 77 features, 15 clases). Ejecutado localmente con
`jupyter nbconvert --execute --inplace` (Python 3.13, PyTorch 2.14 CPU-only). Duración total
aproximada: ~3.5 horas. **Resultado: el notebook corre de punta a punta sin errores y produce
todos los artefactos esperados** en `notebooks/local_run/IDS_Tarea1/results/{parte1,parte2,parte3}/`.

## 1. Errores que surgieron y su solución

| # | Error | Causa | Solución |
|---|-------|-------|----------|
| 1 | `ImportError: cannot import name 'AutoML' from 'flaml'` al inicio de la Parte 2 | El paquete `flaml` estaba instalado sin el extra `[automl]` (el módulo `flaml.automl` no se instala por defecto desde flaml 2.x). El fallback de la celda (`pip install -q flaml`) no soluciona esto porque instala el mismo paquete incompleto. | Se instaló `flaml[automl]` en el entorno y se corrigió la celda de fallback en el notebook (línea con `subprocess.run([...,"pip","install","-q","flaml"...])` → ahora instala `"flaml[automl]"`) para que la instalación automática funcione también en un entorno limpio (ej. Colab). |

No surgió ningún otro `CellExecutionError` en la corrida completa (verificado sobre el log íntegro
de `nbconvert`). El único otro mensaje en el log es un `KeyError` de limpieza de
`joblib.externals.loky.backend.resource_tracker` al cerrar el proceso — es un warning benigno de
liberación de memoria compartida al finalizar, no afecta ninguna celda ni ningún resultado.

### Notas de entorno (no son bugs del notebook, pero valen la pena registrar)
- `optuna` y `sdv` no estaban instalados de entrada y hubo que instalarlos manualmente (el notebook
  no tiene celda de instalación para ellos, a diferencia de `flaml`/`pandas`/`torch`). Considerar
  agregar `optuna` y `sdv` al `%pip install` inicial o a un try/except como el de `flaml`.
- `jupyter`/`nbconvert` no estaban en el `PATH` como comando (`jupyter: command not found`); hubo
  que invocarlos como `python -m nbconvert`. Irrelevante para Colab, solo aplica a ejecución local.

## 2. Resultados clave de la corrida completa

**Dataset:** 2.830.743 filas, 15 clases con desbalance extremo (BENIGN 2.273.097 vs. Heartbleed 11,
Web Attack-Sql Injection 21, Infiltration 36). Split 70/15/15 → train 1.981.517 / val 424.613 / test 424.613.

### Parte 1 — Baselines y DNN manual
| Modelo | macro-F1 (test) | balanced accuracy | tiempo entrenamiento |
|---|---|---|---|
| Regresión Logística | 0.4156 | 0.8626 | 389 s |
| SVM (RBF, submuestra 4000) | 0.3613 | 0.6078 | 0.9 s |
| GBM (misma submuestra, referencia) | 0.6007 | 0.5982 | — |
| DNN manual | **0.6043** | 0.8492 | 41 épocas |

### Parte 2 — AutoML (FLAML) y NAS (Optuna) sobre datos reales
| Modelo | macro-F1 (test) | tiempo | notas |
|---|---|---|---|
| AutoML — `extra_tree` | **0.6860** | 30 s | mejor modelo de toda la corrida |
| DNN + NAS (Optuna, 8 trials) | 0.6028 | búsqueda 2892 s + refit 907 s | arquitectura elegida: 1 capa oculta de 64 |

### Parte 3 — Augmentación sintética (TVAE/SDV) y re-entrenamiento
- **Fidelidad del sintetizador:** score global 0.818 (Column Shapes 0.795, Column Pair Trends 0.840).
- **TSTR (train-synthetic-test-real):** macro-F1 entrenando con sintético = **0.145** vs. macro-F1
  entrenando con real (mismo tamaño de muestra) = **0.351**. El sintético tiene baja utilidad predictiva
  frente al real.
- Augmentación agregó solo 1.223 filas sintéticas sobre 1.981.517 reales (0.06%) — advertencia
  explícita: para **Heartbleed** se pidieron 493 filas sintéticas y solo se pudieron generar 220
  (la clase tiene apenas 11 filas reales para entrenar el TVAE).

| Modelo (train aumentado) | macro-F1 (test) | vs. contraparte real (Parte 2) |
|---|---|---|
| DNN manual | 0.5572 | 0.6043 → **baja 0.047** |
| AutoML — `kneighbor` | 0.5953 | 0.6860 → **baja 0.091** |
| DNN + NAS | 0.5788 | 0.6028 → **baja 0.024** |

## 3. Hallazgos destacables que el markdown base no cubre (sugerencias)

El notebook deja explícitamente como *placeholder* la discusión de la Sección 3.5
("**Discusión (completar/ajustar con los números de la corrida final...)**", celda de la comparación
final) para llenarla con los resultados de la corrida completa. Con los números reales, se sugiere
completar esa celda con lo siguiente — que hoy no está escrito en ningún lado del notebook:

1. **La augmentación sintética empeoró el macro-F1 en los tres modelos, de forma consistente**,
   no solo en algunos. La propia guía de discusión del notebook plantea el escenario "sube en
   algunos y no en otros"; el resultado real es más contundente: bajó en los tres. Combinado con
   el TSTR (0.145 vs 0.351) y el fidelity score moderado (0.82, con Column Shapes más bajo, 0.795),
   la lectura correcta es que el TVAE no capturó bien la señal discriminativa de las clases raras
   y **diluyó** la distribución de entrenamiento en vez de reforzarla. Vale la pena decirlo
   explícitamente como conclusión, no dejarlo abierto.
2. **El presupuesto de cómputo de AutoML/NAS/TVAE no escaló con `fast_dev_run=False`.** El propio
   notebook (notas finales, punto 4) sugiere subir `automl.time_budget_s` a "varios minutos",
   `nas.n_trials` a "30-50" y `tvae_epochs` a "varios cientos" para la corrida completa — pero
   `CONFIG["fast_dev_run"]` solo controla el submuestreo de datos y los epochs del DNN manual; los
   tres parámetros mencionados (`time_budget_s=30`, `n_trials=8`, `tvae_epochs=50`) se mantuvieron
   en sus valores de desarrollo rápido durante toda esta corrida "completa". Es decir: el dataset sí
   es el completo, pero la búsqueda de NAS/AutoML y el entrenamiento del sintetizador siguen siendo
   exploratorios. Si el informe necesita resultados "definitivos" de NAS/AutoML/TVAE, faltaría subir
   estos tres valores en `CONFIG` y volver a correr esas secciones (son las más costosas en tiempo:
   NAS tomó ~48 min de búsqueda + ~15 min de refit por corrida).
3. **AutoML (`extra_tree`, FLAML) fue el mejor modelo de toda la tarea** (macro-F1 0.686), superando
   tanto al DNN manual (0.604) como al DNN+NAS (0.603) — vale la pena resaltarlo como conclusión
   general, ya que contradice la expectativa habitual de que una red neuronal diseñada/optimizada
   supere a un ensemble simple de árboles en este tipo de problema tabular.
4. El desbalance de clases es extremo (Heartbleed: 11 filas de 2.83M). Esto no solo limita el TVAE
   (hallazgo #1) sino que probablemente explica gran parte del techo de macro-F1 (~0.6-0.69) en
   todos los modelos: vale la pena mencionar en el análisis de resultados el recall por clase de las
   categorías ultra-minoritarias (Heartbleed, Web Attack - Sql Injection, Infiltration), no solo el
   macro-F1 agregado — el propio notebook lo sugiere en su guía de discusión pero no llegó a
   completarse con las matrices de confusión reales de esta corrida.
