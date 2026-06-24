# ─── SplatFlow GPU Worker ───────────────────────────────────────────────────
# Base: imagen oficial de Nerfstudio (ya incluye CUDA, COLMAP, ffmpeg, PyTorch)
FROM dromni/nerfstudio:0.3.4

WORKDIR /app

# Dependencias extras para el handler
# Solo runpod + requests. Evitamos el SDK de supabase porque sus pins de
# httpx/pydantic/gotrue chocan con la imagen base de nerfstudio.
RUN pip install --no-cache-dir \
    runpod==1.7.3 \
    requests==2.31.0

COPY handler.py .

# Variables de entorno requeridas en RunPod:
# SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

CMD ["python3", "-u", "handler.py"]
