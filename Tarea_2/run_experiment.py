"""Ejecuta una copia del notebook, conservando los resultados originales."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time


def prepare(profile, phase, workers):
    import nbformat

    root = Path(__file__).resolve().parent
    shared = root / "notebooks/local_run/Tarea2_Deteccion"
    target = shared / "experiments" / profile
    target.mkdir(parents=True, exist_ok=True)
    if not (shared / "splits").exists():
        shutil.copytree(root / "splits", shared / "splits")
    notebook = nbformat.read(root / "notebooks/Tarea_2_Deteccion.ipynb", as_version=4)
    notebook.cells[0].source = (
        f"> Ejecución local: Ryzen 7 7800X3D / RTX 4070 SUPER, perfil `{profile}`. "
        "Las tablas y figuras corresponden a esta corrida; el texto de referencia "
        "del notebook original se conserva y requiere revisión antes de entregar.\n\n"
        + notebook.cells[0].source
    )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        if cell.cell_type != "code":
            continue
        # La focal loss de la CNN puede desbordar fp16 con la escala inicial
        # 65536. Se usa una escala menor y solo avanza el scheduler si hubo paso.
        cell.source = cell.source.replace('GradScaler(DEVICE.type, enabled=USE_AMP)', 'GradScaler(DEVICE.type, enabled=USE_AMP, init_scale=128.0)')
        if 'def fit(' in cell.source:
            cell.source = cell.source.replace('            scaler.step(optimizer)\n            scaler.update()', '            previous_scale = scaler.get_scale()\n            scaler.step(optimizer)\n            scaler.update()')
            cell.source = cell.source.replace('            scheduler.step()\n            if ema is not None:', '            if scaler.get_scale() >= previous_scale:\n                scheduler.step()\n            if ema is not None and scaler.get_scale() >= previous_scale:')
        if 'CONFIG = {' in cell.source:
            override = f'''
# Perfil local RTX 4070 SUPER. Batch efectivo y particiones conservados.
CONFIG["fast_dev_run"] = {profile == 'smoke'!r}
CONFIG["run"] = {{k: k == {phase!r} for k in CONFIG["run"]}}
CONFIG["cnn"]["batch_size"] = 16
CONFIG["yolo_common"]["batch"] = 16
CONFIG["yolo_coco"]["stage_a"]["batch"] = 16
CONFIG["frcnn"]["batch_size"] = 4
CONFIG["frcnn"]["stage_b"]["batch_size"] = 4
CONFIG["eval"]["batch_size"] = {{"cnn": 16, "yolo": 16, "frcnn": 4}}
'''
            if profile == "extended":
                override += 'CONFIG["cnn"]["epochs"] = 120\nCONFIG["yolo_scratch"].update(epochs=200, patience=50, close_mosaic=15)\n'
            cell.source = cell.source.replace('if CONFIG["fast_dev_run"]:', override + '\nif CONFIG["fast_dev_run"]:', 1)
            cell.source = cell.source.replace('workers=2, amp=USE_AMP', f'workers={workers}, amp=USE_AMP')
            if profile == "probe":
                cell.source = cell.source.replace('if CONFIG["fast_dev_run"]:', 'if True:', 1)
                cell.source = cell.source.replace('def set_seed(seed):', 'CONFIG["yolo_common"]["imgsz"] = 640\n\ndef set_seed(seed):', 1)
                # Evita la exportación y mantiene los datos pequeños del smoke.
                cell.source += '\nCONFIG["fast_dev_run"] = True\n'
            cell.source += '\nassert DEVICE.type == "cuda", "Se requiere CUDA para estas corridas"\n'
        if 'SCALE_TAG = ' in cell.source:
            cell.source += f'''
# Datos compartidos; entrenamiento, predicciones y resultados por experimento.
RUNS_DIR = Path({str(target / 'runs')!r})
CKPT_ROOT = RUNS_DIR
PRED_DIR = RUNS_DIR / "predictions"
RESULTS_DIR = Path({str(target / 'results')!r})
for d in (RUNS_DIR, PRED_DIR / "robustness", RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
'''
        if 'export_dir = ' in cell.source:
            cell.source = f'''if not CONFIG["fast_dev_run"]:
    export_dir = Path({str(target / 'entrega')!r})
    shutil.copytree(SPLITS_DIR, export_dir / "splits", dirs_exist_ok=True)
    shutil.copytree(RESULTS_DIR, export_dir / "results", dirs_exist_ok=True)
    print("Resultados independientes en", export_dir)
'''
    return root, target, notebook


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "probe", "baseline", "extended", "selected"], default="smoke")
    parser.add_argument("--phase", choices=["all", "cnn", "yolo_scratch", "yolo_coco", "frcnn", "robustness", "assemble"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    phases = ["cnn", "yolo_scratch", "yolo_coco", "frcnn", "robustness", "assemble"] if args.phase == "all" else [args.phase]
    import nbformat
    from nbclient import NotebookClient

    class LoggedClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg.get("msg_type") == "stream":
                print(msg["content"].get("text", ""), end="", flush=True)
            return super().process_message(msg, cell, cell_index)

    for phase in phases:
        root, target, notebook = prepare(args.profile, phase, args.workers)
        output = target / f"{phase}.ipynb"
        nbformat.write(notebook, output)
        if args.prepare_only:
            print(output, flush=True)
            continue
        status = {"profile": args.profile, "phase": phase, "started": time.time(), "state": "running"}
        status_path = target / "status.json"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Inicio: {args.profile}/{phase}", flush=True)
        def progress(cell, cell_index, **kwargs):
            status["cell"] = cell_index
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"Celda {cell_index}", flush=True)
        client = LoggedClient(notebook, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(root / "notebooks")}}, on_cell_start=progress)
        try:
            client.execute()
            status["state"] = "complete"
        except BaseException as error:
            status.update(state="failed", error=str(error))
            raise
        finally:
            nbformat.write(notebook, output)
            status["finished"] = time.time()
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Terminado: {args.profile}/{phase}", flush=True)


if __name__ == "__main__":
    main()
