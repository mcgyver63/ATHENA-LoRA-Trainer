# 🧠 ATHENA - Entrenament LoRA Multiformat
Framework complet per entrenar models LoRA en Català i Castellà, optimitzat per Qwen 2.5

## 🚀 Flux unificat d’entrenament
```bash
python dataset_creator.py input_pdfs/ Matematiques CA
python trainer.py Matematiques
python merge.py Matematiques
python convert.py Matematiques
python quantize.py Matematiques
```

## 🎯 Característiques principals
- Multiformat de dataset (text / instruction / mixt)
- Multiidioma (Català i Castellà)
- Optimitzat per models petits(4-bit NF4 + LoRA)
- Workflow complet de PDF → Dataset → LoRA → HF → GGUF → Quantitzat
- Anàlisi automàtica d’entrenament
- Totalment local

## 📚 Especialitats disponibles
Matematiques, Fisica, Quimica, Biologia, Informatica, Civil, Automatismes, Robotai

## 📁 Estructura
ATHENA-LoRA-Trainer/
├── README.md
├── README_EN.md
├── requirements.txt
├── dataset_creator.py
├── trainer.py
├── merge.py
├── convert.py
├── quantize.py
├── merge_datasets.py
└── monitor_training.py

## 📦 Dependències
Vegeu requirements.txt

## 🧭 Exemple complet
```bash
python dataset_creator.py input_pdfs/ Biologia ES
python trainer.py Biologia
python merge.py Biologia
python convert.py Biologia
python quantize.py Biologia
```

## 🏆 Filosofia ATHENA
Potenciar el català i castellà en models locals LoRA, per a agents experts totalment offline.

## 📄 Llicència
MIT License
