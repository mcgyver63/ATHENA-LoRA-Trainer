#!/usr/bin/env python3
"""
Quantitza GGUF F16 a Q5_K_M per CPU Windows
"""
import os
import sys
import subprocess
from pathlib import Path

if len(sys.argv) < 2:
    print("Ús: python quantize.py <especialitat>")
    sys.exit(1)

ESPECIALITAT = sys.argv[1]
BASE_PATH = Path("/home/joancarles/lora_training")

# Obtenir nom del model
MODEL_PARENT_DIR = BASE_PATH / "models"
try:
    MODEL_DIR_NAME = next(d.name for d in MODEL_PARENT_DIR.iterdir() if d.is_dir())
except StopIteration:
    raise FileNotFoundError("No s'ha trobat cap carpeta dins de 'models'")

# Paths
F16_FILENAME = f"{MODEL_DIR_NAME}_{ESPECIALITAT}.gguf"
F16_PATH = BASE_PATH / "trained_models" / F16_FILENAME

Q5_FILENAME = f"{MODEL_DIR_NAME}_{ESPECIALITAT}_Q5_K_M.gguf"
Q5_PATH = BASE_PATH / "trained_models" / Q5_FILENAME

QUANTIZE_BIN = "/home/joancarles/llama.cpp/build/bin/llama-quantize"

print(f"Quantitzant {ESPECIALITAT} de F16 a Q5_K_M...")

# Verificar paths
if not F16_PATH.exists():
    print(f"Error: Model F16 no trobat a {F16_PATH}")
    sys.exit(1)

if not Path(QUANTIZE_BIN).exists():
    print(f"Error: llama-quantize no trobat a {QUANTIZE_BIN}")
    print("Compila llama.cpp primer: cd ~/llama.cpp && make")
    sys.exit(1)

# Executar quantització
cmd = [str(QUANTIZE_BIN), str(F16_PATH), str(Q5_PATH), "Q5_K_M"]

print(f"Executant: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, check=True)
    print("\n✅ Quantització completada!")
    print(f"📊 Model original (F16): {F16_PATH.stat().st_size / (1024**3):.2f} GB")
    print(f"📊 Model quantitzat (Q5): {Q5_PATH.stat().st_size / (1024**3):.2f} GB")
    print(f"\n📁 Model per Windows: {Q5_PATH}")
except subprocess.CalledProcessError as e:
    print(f"❌ Error en quantització: {e}")
    sys.exit(1)
