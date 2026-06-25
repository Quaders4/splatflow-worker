"""
SplatFlow — RunPod Serverless Handler
Pipeline: MP4 video → frames → COLMAP → Gaussian Splatting → .ply → Supabase Storage
"""

import os
import subprocess
import tempfile
import time
import requests
import runpod
from pathlib import Path

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SPLAT_BUCKET = "splats"


def download_video(url: str, dest: str) -> None:
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)


def run(cmd: list, timeout: int = 3600, label: str = "") -> None:
    print(f"[handler] {label or ' '.join(cmd[:3])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout[-3000:])  # últimas 3000 chars
    if result.returncode != 0:
        raise RuntimeError(f"{label} falló:\n{result.stderr[-2000:]}")


def upload_to_supabase(local_path: str, storage_path: str) -> str:
    # Sube el archivo a Supabase Storage vía API REST (sin SDK).
    # Docs: https://supabase.com/docs/reference/api/storage
    with open(local_path, "rb") as f:
        data = f.read()

    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SPLAT_BUCKET}/{storage_path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/octet-stream",
        "Cache-Control": "31536000",
        "x-upsert": "false",
    }
    res = requests.post(upload_url, headers=headers, data=data, timeout=300)
    if not res.ok:
        raise RuntimeError(f"Supabase upload falló {res.status_code}: {res.text[:500]}")

    # URL pública (el bucket 'splats' debe ser público)
    return f"{SUPABASE_URL}/storage/v1/object/public/{SPLAT_BUCKET}/{storage_path}"


def handler(job: dict) -> dict:
    job_input = job["input"]
    video_url: str = job_input["video_url"]
    property_id: str = job_input["property_id"]

    print(f"[handler] Iniciando job para propiedad {property_id}")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        video_path = workdir / "input.mp4"
        data_dir   = workdir / "data"
        output_dir = workdir / "outputs"
        export_dir = workdir / "export"

        data_dir.mkdir()
        output_dir.mkdir()
        export_dir.mkdir()

        # ── 1. Descargar video ──────────────────────────────────────────
        print("[handler] Descargando video...")
        download_video(video_url, str(video_path))
        print(f"[handler] Video descargado: {video_path.stat().st_size / 1024 / 1024:.1f} MB")

        # ── 2. Extraer frames y calcular poses con COLMAP ───────────────
        run(
            [
                "ns-process-data", "video",
                "--data", str(video_path),
                "--output-dir", str(data_dir),
                "--num-frames-target", "150",
                "--matching-method", "sequential",
            ],
            timeout=900,
            label="ns-process-data (COLMAP)",
        )

        # ── 3. Validar que COLMAP encontró suficientes poses ───────────
        transforms_file = data_dir / "transforms.json"
        if transforms_file.exists():
            import json as _json
            transforms = _json.loads(transforms_file.read_text())
            n_frames = len(transforms.get("frames", []))
            if n_frames < 30:
                raise RuntimeError(
                    f"Video de baja calidad: COLMAP solo encontró poses para {n_frames} imágenes. "
                    "Graba un video más lento, nítido y orbitando el espacio. Mínimo recomendado: 30 imágenes con pose."
                )
            print(f"[handler] COLMAP encontró {n_frames} imágenes con pose ✓")

        # ── 4. Entrenar modelo Gaussian Splatting ───────────────────────
        run(
            [
                "ns-train", "splatfacto",
                "--data", str(data_dir),
                "--output-dir", str(output_dir),
                "--max-num-iterations", "7000",  # TODO: subir a 30000 para versión final
                "--pipeline.model.cull-alpha-thresh", "0.005",
                "nerfstudio-data",
            ],
            timeout=3600,
            label="ns-train splatfacto",
        )

        # ── 5. Encontrar el config.yml del checkpoint ───────────────────
        configs = sorted(output_dir.glob("splatfacto/*/config.yml"))
        if not configs:
            raise RuntimeError("No se encontró config.yml después del entrenamiento")
        config_path = configs[-1]
        print(f"[handler] Config encontrado: {config_path}")

        # ── 5. Exportar como .ply ───────────────────────────────────────
        run(
            [
                "ns-export", "gaussian-splat",
                "--load-config", str(config_path),
                "--output-dir", str(export_dir),
            ],
            timeout=300,
            label="ns-export gaussian-splat",
        )

        # Buscar el archivo .ply exportado
        ply_files = list(export_dir.glob("*.ply"))
        if not ply_files:
            raise RuntimeError("No se encontró archivo .ply en el directorio de exportación")
        ply_path = ply_files[0]
        print(f"[handler] .ply generado: {ply_path.stat().st_size / 1024 / 1024:.1f} MB")

        # ── 6. Subir a Supabase Storage ─────────────────────────────────
        storage_path = f"{int(time.time())}-{property_id}.ply"
        print(f"[handler] Subiendo a Supabase Storage: {storage_path}")
        splat_url = upload_to_supabase(str(ply_path), storage_path)

        print(f"[handler] ¡Completado! URL: {splat_url}")
        return {
            "splat_url": splat_url,
            "storage_path": storage_path,
        }


runpod.serverless.start({"handler": handler})
