#!/usr/bin/env python3
"""
Merge LoRA Qwen 2.5 Especialitzat
"""
import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if len(sys.argv) < 2:
    print("Ús: python merge.py <especialitat>")
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
MODEL_BASE_PATH = MODEL_PARENT_DIR / MODEL_DIR_NAME
# Generar el nom del directori LoRA d'entrada segons el patró
LORA_INPUT_NAME = f"lora_{MODEL_DIR_NAME}_{ESPECIALITAT}"
LORA_PATH = BASE_PATH / "output" / LORA_INPUT_NAME
# Generar el nom de sortida segons el patró
OUTPUT_PATH_NAME = f"lora_{MODEL_DIR_NAME}_{ESPECIALITAT}_final"
OUTPUT_PATH = BASE_PATH / "output" / OUTPUT_PATH_NAME

print(f"Fusionant LoRA per: {ESPECIALITAT}")

# Verificar paths
if not MODEL_BASE_PATH.exists():
    print(f"Error: Model base no trobat a {MODEL_BASE_PATH}")
    sys.exit(1)

if not LORA_PATH.exists():
    print(f"Error: LoRA no trobat a {LORA_PATH}")
    sys.exit(1)

# Carregar tokenizer
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_BASE_PATH), trust_remote_code=True)

# Carregar model base
base_model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_BASE_PATH),
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# Carregar LoRA
peft_model = PeftModel.from_pretrained(base_model, str(LORA_PATH))

# Fusionar
merged_model = peft_model.merge_and_unload()

# Guardar
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(str(OUTPUT_PATH))
tokenizer.save_pretrained(str(OUTPUT_PATH))

print(f"Fusió completada: {OUTPUT_PATH}")
