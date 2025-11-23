# 🧠 ATHENA - LoRA Training Framework
Complete framework to train LoRA models in Catalan and Spanish, optimized for Qwen 2.5

## 🚀 Unified Workflow
```bash
python dataset_creator.py input_pdfs/ Mathematics CA
python trainer.py Mathematics
python merge.py Mathematics
python convert.py Mathematics
python quantize.py Mathematics
```

## 🎯 Key Features
- Multi-format dataset detection
- Multi-language (Catalan & Spanish)
- Little models optimized (4-bit NF4 + LoRA)
- Full pipeline: PDF → Dataset → LoRA → HF → GGUF → Quantized
- Automatic training analysis
- Fully local and offline friendly

## 📚 Supported Specializations
Mathematics, Physics, Chemistry, Biology, Computer Science, Civil Engineering, Automation, Robotics

## 📁 Structure
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

## 📦 Dependencies
See requirements.txt

## 🧭 Full Example
```bash
python dataset_creator.py input_pdfs/ Biology ES
python trainer.py Biology
python merge.py Biology
python convert.py Biology
python quantize.py Biology
```

## 🏆 ATHENA Philosophy
Strengthening Catalan & Spanish local models and empowering offline expert agents.

## 📄 License
MIT License
