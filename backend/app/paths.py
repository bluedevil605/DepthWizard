import os
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

if os.getenv("VERCEL"):
    RUNTIME_DIR = os.path.join(os.getenv("TMPDIR", "/tmp"), "depthwizard")
else:
    RUNTIME_DIR = STATIC_DIR

OUTPUTS_DIR = os.getenv("DEPTHWIZARD_OUTPUTS_DIR", os.path.join(RUNTIME_DIR, "outputs"))
UPLOADS_DIR = os.getenv("DEPTHWIZARD_UPLOADS_DIR", os.path.join(RUNTIME_DIR, "uploads"))

Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
