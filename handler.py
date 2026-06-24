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
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
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
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    with open(local_path, "rb") as f:
        data = f.read()
    supabase.storage.from_(SPLAT_BUCKET).upload(
        storage_path,
        data,
        file_options={
            "content-type": "application/octet-stream",
            "cache-control": "31536000",
            "upsert": "false",
        },
    )
    url_data = supabase.storage.from_(SPLAT_BUCKET).get_public_url(storage_path)
    return url_data


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

        # ── 3. Entrenar modelo Gaussian Splatting ───────────────────────
        run(
            [
                "ns-train", "splatfacto",
                "--data", str(data_dir),
                "--output-dir", str(output_dir),
                "--max-num-iterations", "30000",
                "--pipeline.model.cull-alpha-thresh", "0.005",
                "nerfstudio-data",
            ],
            timeout=3600,
            label="ns-train splatfacto",
        )

        # ── 4. Encontrar el config.yml del checkpoint ───────────────────
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
