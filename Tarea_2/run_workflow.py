"""Pruebas, corrida base, variantes justificadas por validación y entrega final."""
import json
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
HOME = ROOT / "notebooks/local_run/Tarea2_Deteccion"
EXPERIMENTS = HOME / "experiments"
MODELS = {"cnn": "cnn", "yolo_scratch": "yolo/yolo_scratch", "yolo_coco": "yolo/yolo_coco", "frcnn": "frcnn/stage_b"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(profile, phase):
    state = {"state": "running", "profile": profile, "phase": phase, "updated": time.time(), "pid": os.getpid()}
    (HOME / "workflow_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, "-u", str(ROOT / "run_experiment.py"), "--profile", profile, "--phase", phase], check=True, cwd=ROOT.parent)


def check_losses(profile):
    for path in (EXPERIMENTS / profile / "runs").rglob("*.csv"):
        if path.name not in ("history.csv", "results.csv"):
            continue
        with path.open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                for key, value in row.items():
                    if key.strip().startswith(("train_", "train/")) and not math.isfinite(float(value)):
                        raise RuntimeError(f"Pérdida no finita en {path}: {key}={value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-smoke", action="store_true", help="Solo si el smoke ya terminó correctamente")
    args = parser.parse_args()
    from filelock import FileLock
    HOME.mkdir(parents=True, exist_ok=True)
    with FileLock(str(HOME / "workflow.lock"), timeout=0):
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            if args.skip_smoke:
                smoke = read(EXPERIMENTS / "smoke/status.json")
                if smoke.get("state") != "complete" or smoke.get("phase") != "assemble":
                    raise RuntimeError("El smoke completo debe terminar antes de usar --skip-smoke")
            if not args.skip_smoke:
                run("smoke", "all")
            # CNN y Faster R-CNN ya usaron resolución y batches reales en smoke.
            # Solo YOLO reducía la resolución a 320 en ese perfil.
            for phase in ("yolo_scratch", "yolo_coco"):
                run("probe", phase)
            check_losses("smoke")
            check_losses("probe")
            for model in MODELS:
                run("baseline", model)
                check_losses("baseline")
            # Predefinido: ampliar solo si llegó al presupuesto y su mejor val
            # está en el último cuarto. Nunca se usa test para esta decisión.
            candidates = {}
            for model, budget in (("cnn", 80), ("yolo_scratch", 90)):
                summary = read(EXPERIMENTS / "baseline/runs" / MODELS[model] / "train_summary.json")
                extend = summary["epochs_run"] >= budget and summary["best_epoch"] >= 0.75 * budget
                candidates[model] = {"extend": extend, "baseline_val": summary["best_val_map"]}
                if extend:
                    run("extended", model)
            selected = EXPERIMENTS / "selected"
            # Copia reproducible de las variantes elegidas. Robustez se recalcula.
            shutil.copytree(EXPERIMENTS / "baseline/runs", selected / "runs", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("robustness"))
            winners = {model: "baseline" for model in MODELS}
            for model, info in candidates.items():
                if not info["extend"]:
                    continue
                extended = read(EXPERIMENTS / "extended/runs" / MODELS[model] / "train_summary.json")
                info["extended_val"] = extended["best_val_map"]
                if extended["best_val_map"] <= info["baseline_val"]:
                    continue
                winners[model] = "extended"
                source = EXPERIMENTS / "extended/runs"
                shutil.copytree(source / MODELS[model], selected / "runs" / MODELS[model], dirs_exist_ok=True)
                for path in (source / "predictions").glob(f"{model}_*.parquet"):
                    shutil.copy2(path, selected / "runs/predictions" / path.name)
                shutil.copy2(source / "efficiency" / f"{model}.json", selected / "runs/efficiency" / f"{model}.json")
            (selected / "selection.json").write_text(json.dumps({"winners": winners, "validation": candidates}, indent=2), encoding="utf-8")
            run("selected", "robustness")
            run("selected", "assemble")
            state = {"state": "complete", "updated": time.time(), "winners": winners}
        except BaseException as error:
            state = {"state": "failed", "updated": time.time(), "error": str(error)}
            raise
        finally:
            (HOME / "workflow_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
