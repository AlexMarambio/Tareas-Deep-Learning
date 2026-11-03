"""Descarga reanudable por segmentos; valida el MD5 publicado antes de usar el ZIP."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request

URL = "https://zenodo.org/api/records/17950781/files/FjordFish.zip/content"
SIZE = 6058780342
MD5 = "951bf139c3e412fa7924fa7ffc34a307"
ROOT = Path(__file__).resolve().parent / "notebooks/local_run/Tarea2_Deteccion/work/raw"
CHUNK = 128 * 1024 * 1024


def segment(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    path = ROOT / "segments" / f"{index:04d}.part"
    failures = 0
    while failures < 12:
        done = path.stat().st_size if path.exists() else 0
        if done == end - start + 1:
            return path
        try:
            # Rangos acotados evitan conexiones grandes estancadas en Zenodo.
            request_end = min(end, start + done + 4 * 1024 * 1024 - 1)
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start + done}-{request_end}"})
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start + done}-"):
                    raise RuntimeError("El servidor no respetó el rango solicitado")
                with path.open("ab") as output:
                    shutil.copyfileobj(response, output, 1024 * 1024)
            failures = 0
        except Exception:
            failures += 1
            if failures == 12:
                raise
            time.sleep(min(60, 2 ** failures))
    raise RuntimeError(f"Segmento incompleto: {index}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / "FjordFish.zip"
    if output.exists():
        with output.open("rb") as stream:
            if hashlib.file_digest(stream, "md5").hexdigest() == MD5:
                print("ZIP existente verificado", flush=True)
                return
        raise RuntimeError("ZIP existente con checksum incorrecto")
    (ROOT / "segments").mkdir(exist_ok=True)
    count = (SIZE + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, task in enumerate(as_completed([pool.submit(segment, i) for i in range(count)]), 1):
            task.result()
            print(f"Segmentos: {done}/{count}", flush=True)
    temporary = ROOT / "FjordFish.assembling"
    digest = hashlib.md5()
    with temporary.open("wb") as stream:
        for i in range(count):
            with (ROOT / "segments" / f"{i:04d}.part").open("rb") as part:
                while block := part.read(1024 * 1024):
                    digest.update(block)
                    stream.write(block)
    if digest.hexdigest() != MD5:
        raise RuntimeError("El MD5 de la descarga no coincide")
    temporary.replace(output)
    print(f"ZIP verificado: {output}", flush=True)


if __name__ == "__main__":
    main()
