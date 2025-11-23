#!/usr/bin/env python3
"""
Converteix GGUF F16
"""
import os
import sys
import subprocess
from pathlib import Path

if len(sys.argv) < 2:
    print("Ús: python convert.py <especialitat>")
    sys.exit(1)

ESPECIALITAT = sys.argv[1]
BASE_PATH = Path("/home/joancarles/lora_training")
# Obtenir el nom del model de la carpeta 'models'
# Això assumeix que només hi ha una carpeta dins de 'models'
MODEL_PARENT_DIR = BASE_PATH / "models"
try:
    MODEL_DIR_NAME = next(d.name for d in MODEL_PARENT_DIR.iterdir() if d.is_dir())
except StopIteration:
    raise FileNotFoundError("No s'ha trobat cap carpeta dins de 'models'")
# Generar el nom del directori del model fusionat d'entrada segons el patró
MERGED_DIR_NAME = f"lora_{MODEL_DIR_NAME}_{ESPECIALITAT}_final"
MERGED_PATH = BASE_PATH / "output" / MERGED_DIR_NAME
# Generar el nom del fitxer de sortida .gguf segons el patró
GGUF_FILENAME = f"{MODEL_DIR_NAME}_{ESPECIALITAT}.gguf"
OUTPUT_PATH = BASE_PATH / "trained_models" / GGUF_FILENAME
CONVERTER_SCRIPT = "/home/joancarles/llama.cpp/convert_hf_to_gguf.py"

print(f"Convertint {ESPECIALITAT} a GGUF F16...")

# Verificar paths
if not MERGED_PATH.exists():
    print(f"Error: Model fusionat no trobat a {MERGED_PATH}")
    sys.exit(1)

if not Path(CONVERTER_SCRIPT).exists():
    print(f"Error: Converter no trobat a {CONVERTER_SCRIPT}")
    sys.exit(1)

# Crear directori destí
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Executar conversió
cmd = [
    "python",
    CONVERTER_SCRIPT,
    str(MERGED_PATH),
    "--outfile",
    str(OUTPUT_PATH),
    "--outtype",
    "f16",
]

print(f"Executant: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("Conversió completada!")
    print(f"Model GGUF: {OUTPUT_PATH}")
except subprocess.CalledProcessError as e:
    print(f"Error en conversió: {e}")
    print(f"STDERR: {e.stderr}")
    sys.exit(1)
