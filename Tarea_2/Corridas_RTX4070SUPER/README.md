# Corridas de Tarea 2 en RTX 4070 SUPER

Esta carpeta contiene los resultados de los experimentos ejecutados el 3 de
noviembre de 2026 en el PC de Diego: Ryzen 7 7800X3D y NVIDIA RTX 4070 SUPER de
12 GB. Incluye notebooks con salidas, pesos, checkpoints para reanudar,
predicciones, métricas, historiales CSV, figuras y registros. La ejecución
automática terminó correctamente (`workflow_status.json`: `complete`).

## Qué abrir

- **[selected/assemble.ipynb](selected/assemble.ipynb)**: notebook final con las
  salidas de los cuatro modelos elegidos y la evaluación de robustez. Verificado
  sin celdas con errores de ejecución.
- **[selected/entrega/](selected/entrega/)**: particiones, tablas y figuras finales.
- **[Tabla comparativa final](selected/results/comparacion/comparison_table.csv)**.
- **[selected/selection.json](selected/selection.json)**: variantes elegidas y
  resultados de validación usados para elegirlas.

## Resultados finales

Las variantes se eligen con validación, sin usar test para decidir. Los números
de esta tabla corresponden a **test** y al evaluador común del notebook.

| Modelo             | Variante elegida | Épocas ejecutadas |   mAP50-95 |      mAP50 |         F1 |
| ------------------ | ---------------- | ----------------: | ---------: | ---------: | ---------: |
| CNN propia         | Extended         |               110 |     0,1248 |     0,2597 |     0,5239 |
| YOLO11s desde cero | Extended         |               200 |     0,2238 |     0,3955 |     0,6071 |
| YOLO11s COCO       | Baseline         |           10 + 30 | **0,2971** | **0,4393** | **0,6730** |
| Faster R-CNN COCO  | Baseline         |            4 + 10 |     0,2258 |     0,3690 |     0,6098 |

CNN extended tenía un máximo de 120 épocas y se detuvo en la 110 por early
stopping; su mejor checkpoint fue el de la época 80. YOLO extended completó 200
épocas y su mejor checkpoint fue el de la 181. Las tablas usan los mejores pesos,
no necesariamente los de la última época.

En baseline, CNN obtuvo mAP50-95 de test 0,1166 y YOLO desde cero 0,1844.
Las corridas largas mejoraron a 0,1248 y 0,2238, respectivamente. El mejor modelo
por validación para la prueba de robustez fue YOLO COCO: se evaluaron cinco
degradaciones con tres niveles, además de comparar los cuatro modelos en el
nivel severo. Los CSV están en `selected/results/parte3/`.

## Contenido

| Carpeta              | Contenido                                                                                                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline/`          | Cuatro modelos completos: CNN 80 épocas, YOLO desde cero 90, YOLO COCO 10+30 y Faster R-CNN 4+10.                                             |
| `extended/`          | CNN con presupuesto de 120 épocas y YOLO desde cero con 200, entrenados desde cero en corridas separadas.                                     |
| `selected/`          | Copia de los modelos ganadores, robustez final, figuras y notebook ensamblado.                                                                |

Dentro de cada experimento, `runs/` contiene pesos (`best.pt`), checkpoints
(`last.pt`), resúmenes y predicciones; `results/` contiene tablas y figuras.
`status.json` describe la última fase ejecutada en ese experimento.
`workflow.log` y `workflow.err.log` son los registros de la secuencia automática;
el segundo también contiene advertencias de las bibliotecas, aunque la ejecución
haya terminado correctamente.

Los directorios `selected/` y `selected/entrega/` repiten parte del contenido de
otras carpetas para conservar la estructura generada por el notebook.
No se incluyen el dataset, `.venv`, pesos, checkpoints ni pruebas de diagn?stico.

## Entorno y receta

- Python 3.13.3; PyTorch 2.14.1+cu126; torchvision 0.29.1+cu126.
- Ultralytics 8.4.177; torchmetrics 1.9.0; faster-coco-eval 1.8.0.
- AMP activado. Batch CNN 16 (efectivo 16), YOLO 16, Faster R-CNN 4 (efectivo 4,
  en ambas etapas). Las corridas del workflow usan hasta cuatro workers de YOLO.
- Misma partición del repositorio: 2101 imágenes de train, 451 de val y 450 de
  test. Lado largo de trabajo 640; entrada de CNN por letterbox a 384×640.
- Se ampliaron los modelos desde cero cuando llegaron al presupuesto base y su
  mejor validación quedó en el último cuarto de las épocas. Después se eligió
  entre baseline y extended por validación dentro de cada familia. Los valores
  nativos de validación de Ultralytics pueden diferir de los del evaluador común
  de las tablas finales.
- En los loops propios se redujo la escala inicial de GradScaler a 128; scheduler
  y EMA solo avanzan cuando se realiza el paso del optimizador. Los logs usan
  UTF-8. El notebook fuente original se conserva.

## Reproducir las corridas

Los pesos y checkpoints se conservan localmente y no se publican en GitHub.
No se necesita Git LFS para descargar estos resultados.

Los notebooks de esta carpeta conservan las rutas locales de la ejecución.
Para repetir las corridas se deben usar los ejecutores del repositorio, desde
su raíz, siguiendo [CORRIDAS_RTX.md](../CORRIDAS_RTX.md). Las dependencias están en
`Tarea_2/requirements-rtx.txt` y las versiones instaladas en
`Tarea_2/requirements-rtx-lock.txt`.

```powershell
.\Tarea_2\.venv\Scripts\python.exe -u .\Tarea_2\run_workflow.py
```

Se requiere el dataset FjordFish original o el `processed.zip` generado por el
notebook. El ejecutor escribe nuevas salidas en `Tarea_2/notebooks/local_run/`;
esta carpeta subida al repositorio es la copia de los resultados ya obtenidos.

**Antes de entregar la tarea:** revisar los textos explicativos del notebook.
Se regeneraron las salidas, pero el texto de referencia conserva menciones al
hardware y las corridas anteriores. Esta carpeta documenta el trabajo de
ejecución y no sustituye esa revisión académica.
