# Corridas en Ryzen 7 7800X3D / RTX 4070 SUPER

El notebook original y los resultados de Alex se conservan. El ejecutor genera
copias sin salidas y aplica la configuración de GPU antes de inicializar CUDA.
Usa las particiones originales de `splits/` y comparte solo los datos entre perfiles.

## Entorno (PowerShell, desde la raíz del repositorio)

```powershell
python -m venv Tarea_2/.venv
Tarea_2/.venv/Scripts/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
Tarea_2/.venv/Scripts/python -m pip install -r Tarea_2/requirements-rtx.txt
Tarea_2/.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Ejecución

```powershell
python -u Tarea_2/download_dataset.py
Tarea_2/.venv/Scripts/python -u Tarea_2/run_experiment.py --profile smoke
Tarea_2/.venv/Scripts/python -u Tarea_2/run_experiment.py --profile baseline
```

`run_workflow.py` encadena smoke, probe a 640, baseline y variantes largas
justificadas por validación. La regla se fija antes de entrenar: ampliar solo
si se alcanzó el presupuesto y el mejor checkpoint está en el último cuarto.
Elige entre baseline y extended por val mAP50-95 y reúne los ganadores en
`experiments/selected/`, donde recalcula robustez y ensambla el notebook final.
`workflow_status.json` indica la fase actual o el error que detuvo la secuencia.
`requirements-rtx-lock.txt` registra las versiones instaladas.

Cada modelo se ejecuta en un kernel nuevo. Una fase fallida detiene la secuencia.
Para retomar, repetir el comando: los checkpoints del notebook permiten continuar.
También se puede elegir `--phase cnn`, `yolo_scratch`, `yolo_coco`, `frcnn`,
`robustness` o `assemble`. No cambiar configuración mientras haya un entrenamiento
incompleto. `--prepare-only` genera las copias sin ejecutar.

Los perfiles usan batch CNN 16 (efectivo 16), YOLO 16, Faster R-CNN 4 en ambas
etapas (efectivo 4), cuatro workers de YOLO y AMP automático. Los DataLoader
propios mantienen cero workers en Windows. Baseline conserva las épocas 80,
90, 10+30 y 4+10. El smoke reduce muestras, épocas y resolución YOLO; se debe
comprobar memoria a resolución real al comenzar baseline.

En las copias ejecutables se usa escala AMP inicial 128 para los loops propios:
la escala predeterminada 65536 omitía los primeros pasos de la CNN por overflow.
El scheduler y EMA avanzan solo si GradScaler realizó la actualización. Los logs
usan UTF-8 para soportar caracteres de Ultralytics en Windows. Estos cambios
están en el ejecutor; el notebook original permanece como referencia.

Las salidas están en `notebooks/local_run/Tarea2_Deteccion/experiments/<perfil>/`:
notebooks por fase, `runs/`, `results/`, `entrega/` y `status.json`. La fase
assemble genera el notebook completo con todas las salidas. No reemplaza
automáticamente las figuras originales del repositorio.

## Variantes después de revisar validación

`--profile extended --phase cnn` usa 120 épocas; `--profile extended --phase
yolo_scratch` usa 200 épocas, paciencia 50 y close_mosaic 15. Ambas empiezan
desde cero en carpetas independientes. Ejecutarlas solo si la evolución de
validación de baseline justifica más entrenamiento. Comparar por mAP50-95 de
validación; reservar test para la evaluación de las variantes seleccionadas.

La descarga usa ocho conexiones y segmentos reanudables, y verifica el MD5 antes
de publicar `FjordFish.zip`. Si se dispone de `processed.zip`, dejarlo en
`notebooks/local_run/Tarea2_Deteccion/data/processed.zip` permite omitir el ZIP
original. Los segmentos se conservan para poder reanudar.
