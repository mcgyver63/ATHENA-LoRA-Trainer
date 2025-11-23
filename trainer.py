#!/usr/bin/env python3
"""
Trainer Optimitzat ATHENA - Versió MULTIFORMAT
Compatible amb tots els formats de dataset (Català/Castellà)
"""
import os
import sys
import torch
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from termcolor import colored
import warnings

warnings.filterwarnings("ignore")

# Optimitzacions CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def print_header():
    """Mostra header ATHENA"""
    print(colored("=" * 70, "cyan"))
    print(
        colored(
            "   🧠 ATHENA TRAINER v3.0 - MULTIFORMAT (Català/Castellà)",
            "cyan",
            attrs=["bold"],
        )
    )
    print(colored("=" * 70, "cyan"))


def detectar_formato_dataset(data):
    """
    Detecta automàticament el format del dataset
    Retorna: 'text' (català), 'instruction' (castellà), o 'mixt'
    """
    if not data or len(data) == 0:
        return "desconegut"

    primer_item = data[0]

    # Comptem quins camps estan presents
    tiene_text = "text" in primer_item
    tiene_instruction = "instruction" in primer_item
    tiene_output = "output" in primer_item

    # Anàlisi detallat del format
    if tiene_text and not tiene_instruction:
        # Format original català
        return "text"
    elif tiene_instruction and tiene_output and not tiene_text:
        # Format dataset_creatorB.py (castellà)
        return "instruction"
    elif tiene_text and tiene_instruction:
        # Format mixt
        return "mixt"
    else:
        # Intentem detectar pel contingut
        if "### Instruction:" in str(primer_item):
            return "text"
        elif "instruction" in str(primer_item).lower():
            return "instruction"
        else:
            return "desconegut"


def convertir_a_formato_unificado(data, formato_detectado):
    """
    Converteix qualsevol format al format unificat per l'entrenament
    """
    textos_unificados = []

    for item in data:
        if formato_detectado == "text":
            # Format original: ja té el text en format instruction-response
            textos_unificados.append(item["text"])

        elif formato_detectado == "instruction":
            # Format dataset_creatorB.py: cal construir el text
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            # Construim el text en format instruction-response
            if input_text:
                texto_unificado = f"### Instruction:\n{instruction}\n{input_text}\n\n### Response:\n{output}"
            else:
                texto_unificado = (
                    f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                )

            textos_unificados.append(texto_unificado)

        elif formato_detectado == "mixt":
            # Format mixt: prioritzem 'text' si existeix, sinó construïm
            if "text" in item and item["text"]:
                textos_unificados.append(item["text"])
            elif "instruction" in item and "output" in item:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output = item.get("output", "")

                if input_text:
                    texto_unificado = f"### Instruction:\n{instruction}\n{input_text}\n\n### Response:\n{output}"
                else:
                    texto_unificado = (
                        f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    )

                textos_unificados.append(texto_unificado)

        else:
            # Format desconegut: intentem extraure qualsevol text
            if "text" in item:
                textos_unificados.append(item["text"])
            elif "content" in item:
                textos_unificados.append(item["content"])
            else:
                # Últim recurs: convertim tot l'item a string
                textos_unificados.append(str(item))

    return textos_unificados


def validar_dataset(data, formato_detectado):
    """
    Valida que el dataset tingui la qualitat mínima necessària
    """
    if not data:
        return False, "Dataset buit"

    textos = convertir_a_formato_unificado(data, formato_detectado)

    # Estadístiques de validació
    longitudes = [len(texto.strip()) for texto in textos]
    textos_vacios = sum(1 for l in longitudes if l == 0)
    textos_cortos = sum(1 for l in longitudes if 0 < l < 50)
    textos_adecuados = sum(1 for l in longitudes if 50 <= l <= 2000)

    print(colored("  📊 Estadístiques del dataset:", "blue"))
    print(f"    • Total ítems: {len(data)}")
    print(f"    • Texts buits: {textos_vacios}")
    print(f"    • Texts massa curts (<50 chars): {textos_cortos}")
    print(f"    • Texts adequats: {textos_adecuados}")

    # Critèris de qualitat
    if textos_vacios > len(data) * 0.1:  # Més del 10% buits
        return False, f"Massos texts buits: {textos_vacios}/{len(data)}"

    if textos_adecuados < len(data) * 0.5:  # Menys del 50% adequats
        return False, f"Pocs texts adequats: {textos_adecuados}/{len(data)}"

    return True, "Dataset vàlid"


def analitzar_resultats(trainer, output_dir, temps_total):
    """Analitza els resultats després de l'entrenament"""
    print(colored("\n📊 ANÀLISI DE RESULTATS", "cyan", attrs=["bold"]))
    print("=" * 60)

    # Temps d'entrenament
    hores = int(temps_total // 3600)
    minuts = int((temps_total % 3600) // 60)
    segons = int(temps_total % 60)
    print(colored(f"⏱️  Temps total: {hores}h {minuts}m {segons}s", "green"))

    if not trainer.state.log_history:
        print(colored("⚠️  No hi ha històric de logs!", "yellow"))
        return

    # Extreure mètriques
    train_losses = []
    learning_rates = []
    steps = []

    for entry in trainer.state.log_history:
        if "loss" in entry:
            train_losses.append(entry["loss"])
            steps.append(entry.get("step", len(train_losses)))
        if "learning_rate" in entry:
            learning_rates.append(entry["learning_rate"])

    if train_losses:
        # Estadístiques
        print(colored("\n📈 EVOLUCIÓ DE L'ENTRENAMENT:", "yellow"))
        print(f"  • Loss inicial: {train_losses[0]:.4f}")
        print(f"  • Loss final: {train_losses[-1]:.4f}")
        millora = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"  • Millora total: {millora:.1f}%")

        # Detecció de problemes
        print(colored("\n🔍 DIAGNÒSTIC:", "yellow"))

        # Verificar convergència
        if len(train_losses) > 10:
            recent_losses = train_losses[-10:]
            variance = np.var(recent_losses)

            if variance < 0.001:
                print(colored("  ✅ Model ben convergit", "green"))
            elif variance < 0.01:
                print(colored("  ⚠️  Convergència acceptable", "yellow"))
            else:
                print(colored("  ❌ Model inestable o no convergit", "red"))

        # Verificar loss final
        if train_losses[-1] < 0.5:
            print(colored("  ✅ Loss excel·lent (<0.5)", "green"))
        elif train_losses[-1] < 1.0:
            print(colored("  ✅ Loss bo (<1.0)", "green"))
        elif train_losses[-1] < 1.5:
            print(colored("  ⚠️  Loss acceptable (<1.5)", "yellow"))
        else:
            print(colored("  ❌ Loss alt (>1.5) - Necessita més entrenament", "red"))

        # Verificar millora
        if len(train_losses) > 20:
            millora_recent = abs(train_losses[-20] - train_losses[-1])
            if millora_recent < 0.01:
                print(
                    colored(
                        "  ⚠️  Possible estancament - Considera ajustar LR", "yellow"
                    )
                )

        # Crear gràfics
        print(colored("\n📊 Generant gràfics...", "blue"))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Gràfic 1: Loss
        axes[0].plot(steps, train_losses, "b-", alpha=0.7, linewidth=2)
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Evolució del Training Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_facecolor("#f0f0f0")

        # Marcar millor i pitjor punt
        min_idx = np.argmin(train_losses)
        max_idx = np.argmax(train_losses)
        axes[0].plot(
            steps[min_idx],
            train_losses[min_idx],
            "go",
            markersize=10,
            label=f"Millor: {train_losses[min_idx]:.3f}",
        )
        axes[0].plot(
            steps[max_idx],
            train_losses[max_idx],
            "ro",
            markersize=10,
            label=f"Pitjor: {train_losses[max_idx]:.3f}",
        )
        axes[0].legend()

        # Gràfic 2: Learning Rate
        if learning_rates:
            axes[1].plot(learning_rates, "orange", alpha=0.7, linewidth=2)
            axes[1].set_xlabel("Steps")
            axes[1].set_ylabel("Learning Rate")
            axes[1].set_title("Learning Rate Schedule")
            axes[1].grid(True, alpha=0.3)
            axes[1].set_facecolor("#f0f0f0")

        plt.suptitle(
            f"ATHENA Training Analysis - {output_dir.name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Guardar gràfic
        plot_path = output_dir / "training_analysis.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        print(colored(f"  ✅ Gràfics guardats a: {plot_path}", "green"))
        plt.close()

    # Guardar resum en JSON
    summary = {
        "model": output_dir.name,
        "timestamp": datetime.now().isoformat(),
        "training_time_seconds": temps_total,
        "training_time_formatted": f"{hores}h {minuts}m {segons}s",
        "final_loss": train_losses[-1] if train_losses else None,
        "initial_loss": train_losses[0] if train_losses else None,
        "improvement_percent": millora if train_losses else None,
        "total_steps": steps[-1] if steps else None,
        "convergence_variance": float(variance) if "variance" in locals() else None,
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(colored(f"\n📄 Resum guardat a: {summary_path}", "green"))


def test_rapid_model(model, tokenizer, especialitat):
    """Test ràpid del model entrenat"""
    print(colored("\n🧪 TEST RÀPID DEL MODEL", "cyan", attrs=["bold"]))
    print("=" * 60)

    # Prompts de test segons especialitat
    test_prompts = {
        "Matematiques": [
            "### Instruction:\nResol l'equació: 2x + 5 = 13\n\n### Response:",
            "### Instruction:\nCalcula l'àrea d'un cercle de radi 5.\n\n### Response:",
        ],
        "Fisica": [
            "### Instruction:\nExplica la segona llei de Newton.\n\n### Response:",
            "### Instruction:\nQuè és l'energia cinètica?\n\n### Response:",
        ],
        "Quimica": [
            "### Instruction:\nQuè és una reacció exotèrmica?\n\n### Response:",
            "### Instruction:\nExplica què és el pH.\n\n### Response:",
        ],
        "Biologia": [
            "### Instruction:\nQuè és la fotosíntesi?\n\n### Response:",
            "### Instruction:\nExplica l'estructura de l'ADN.\n\n### Response:",
        ],
        "Informatica": [
            "### Instruction:\nQuè és un algoritme?\n\n### Response:",
            "### Instruction:\nExplica què és la complexitat O(n).\n\n### Response:",
        ],
        "Civil": [
            "### Instruction:\nQuè és la resistència a compressió?\n\n### Response:",
            "### Instruction:\nExplica què és un pont en cantilever.\n\n### Response:",
        ],
        "Automatismes": [
            "### Instruction:\nQuè és un PLC?\n\n### Response:",
            "### Instruction:\nExplica què és un sensor inductiu.\n\n### Response:",
        ],
        "Robotai": [
            "### Instruction:\nQuè és la cinemàtica inversa?\n\n### Response:",
            "### Instruction:\nExplica què és el machine learning.\n\n### Response:",
        ],
    }

    prompts = test_prompts.get(
        especialitat,
        [
            "### Instruction:\nExplica un concepte bàsic del teu domini.\n\n### Response:"
        ],
    )

    model.eval()

    for i, prompt in enumerate(prompts[:2], 1):  # Només 2 tests per rapidesa
        print(colored(f"\n📝 Test {i}:", "yellow"))
        print(f"Prompt: {prompt[:80]}...")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                min_new_tokens=20,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        print(colored(f"🤖 Resposta: {response[:200]}...", "green"))

        # Avaluació ràpida
        words = response.split()
        if len(words) < 10:
            print(colored("  ⚠️ Resposta massa curta", "yellow"))
        elif len(words) > 80:
            print(colored("  ⚠️ Resposta massa llarga", "yellow"))
        else:
            print(colored("  ✅ Longitud adequada", "green"))


def main():
    print_header()

    if len(sys.argv) < 2:
        print(colored("❌ Error: Falta especificar l'especialitat", "red"))
        print(colored("Ús: python trainer.py <especialitat>", "yellow"))
        print(
            colored(
                "Especialitats: Matematiques, Fisica, Quimica, Biologia, Informatica, Civil, Automatismes, Robotai",
                "cyan",
            )
        )
        sys.exit(1)

    ESPECIALITAT = sys.argv[1]
    BASE_PATH = Path("/home/joancarles/lora_training")

    # Obtenir model
    MODEL_PARENT_DIR = BASE_PATH / "models"
    try:
        MODEL_DIR_NAME = next(d.name for d in MODEL_PARENT_DIR.iterdir() if d.is_dir())
    except StopIteration:
        print(colored("❌ No s'ha trobat cap carpeta dins de 'models'", "red"))
        sys.exit(1)

    MODEL_PATH = MODEL_PARENT_DIR / MODEL_DIR_NAME
    DATASET_PATH = BASE_PATH / "datasets" / f"dataset_{ESPECIALITAT}.json"
    OUTPUT_DIR_NAME = f"lora_{MODEL_DIR_NAME}_{ESPECIALITAT}"
    OUTPUT_DIR = BASE_PATH / "output" / OUTPUT_DIR_NAME

    # Configuració LoRA optimitzada per multiformat
    LORA_CONFIG = {
        "r": 64,  # Rank alt per múltiples idiomes
        "lora_alpha": 16,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }

    print(colored(f"\n📚 Configuració:", "cyan"))
    print(f"  • Especialitat: {colored(ESPECIALITAT, 'green', attrs=['bold'])}")
    print(f"  • Model: {MODEL_PATH.name}")
    print(f"  • Dataset: {DATASET_PATH.name}")
    print(f"  • Output: {OUTPUT_DIR.name}")

    # Verificacions
    if not MODEL_PATH.exists():
        print(colored(f"❌ Model no trobat a {MODEL_PATH}", "red"))
        sys.exit(1)

    if not DATASET_PATH.exists():
        print(colored(f"❌ Dataset no trobat a {DATASET_PATH}", "red"))
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Carregar tokenizer
    print(colored("\n🔄 Carregant tokenizer...", "blue"))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important per generació

    # Carregar i processar dataset
    print(colored("📚 Carregant dataset...", "blue"))
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # DETECTAR FORMAT DEL DATASET
    formato_detectado = detectar_formato_dataset(data)
    print(colored(f"  • Format detectat: {formato_detectado}", "green"))

    # VALIDAR DATASET
    es_valido, mensaje_validacion = validar_dataset(data, formato_detectado)
    if not es_valido:
        print(colored(f"❌ Error en dataset: {mensaje_validacion}", "red"))
        sys.exit(1)

    # CONVERTIR A FORMAT UNIFICAT
    print(colored("🔄 Convertint a format unificat...", "blue"))
    textos_unificados = convertir_a_formato_unificado(data, formato_detectado)

    # Mostrar exemples
    print(colored("  📝 Exemple de format unificat:", "cyan"))
    for i, texto in enumerate(textos_unificados[:1]):  # Mostrar primer exemple
        print(f"    {texto[:100]}...")

    # Crear dataset
    dataset = Dataset.from_dict({"text": textos_unificados})

    # NO fem split train/eval - tot per entrenar!
    print(colored(f"  • Total exemples: {len(dataset)}", "green"))

    # Funció de tokenització optimitzada
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    # Tokenitzar tot el dataset
    print(colored("🔄 Tokenitzant dataset...", "blue"))
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    # Carregar model amb quantització
    print(colored("🤖 Carregant model base...", "blue"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Preparar per entrenament
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Aplicar LoRA
    print(colored("🔧 Aplicant LoRA...", "blue"))
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    # Mostrar informació del model
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(colored(f"  • Paràmetres entrenables: {trainable_params:,}", "green"))
    print(colored(f"  • Paràmetres totals: {total_params:,}", "green"))
    print(
        colored(
            f"  • Percentatge entrenable: {trainable_params/total_params*100:.2f}%",
            "green",
        )
    )

    # Data collator simple
    class SimpleDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            batch = {
                "input_ids": torch.tensor(
                    [f["input_ids"] for f in features], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    [f["attention_mask"] for f in features], dtype=torch.long
                ),
                "labels": torch.tensor(
                    [f["labels"] for f in features], dtype=torch.long
                ),
            }
            return batch

    data_collator = SimpleDataCollator(tokenizer)

    # Configuració d'entrenament SENSE EVAL
    print(colored("\n⚙️ Configurant entrenament...", "blue"))

    # Calcular steps
    batch_size = 3
    gradient_accumulation = 3
    effective_batch_size = batch_size * gradient_accumulation
    num_epochs = 3
    total_steps = (len(tokenized_dataset) // effective_batch_size) * num_epochs

    print(f"  • Batch size efectiu: {effective_batch_size}")
    print(f"  • Steps totals estimats: {total_steps}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=2e-4,
        weight_decay=0.01,
        # SENSE EVALUATION per estalviar temps!
        eval_strategy="no",  # No avaluació
        save_strategy="steps",
        save_steps=500,  # Guardar cada 500 steps
        save_total_limit=3,
        logging_steps=10,
        logging_first_step=True,
        # Optimitzacions
        bf16=True,
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        # Altres
        report_to="none",  # No tensorboard per velocitat
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        # Important per continuar
        load_best_model_at_end=False,  # No hi ha eval
        resume_from_checkpoint=True,  # Automàtic si existeix
    )

    # Buscar checkpoint existent
    checkpoints = sorted(
        [
            d
            for d in OUTPUT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
    )

    if checkpoints:
        last_checkpoint = checkpoints[-1]
        print(
            colored(
                f"\n🔄 CONTINUANT des de: {last_checkpoint.name}",
                "yellow",
                attrs=["bold"],
            )
        )
        resume_from = str(last_checkpoint)
    else:
        print(colored("\n🆕 INICIANT entrenament nou", "green", attrs=["bold"]))
        resume_from = None

    # Crear trainer sense eval_dataset
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        eval_dataset=None,  # Sense eval dataset!
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_seq_length=512,
        dataset_text_field="text",
    )

    print(colored("\n🚀 INICIANT ENTRENAMENT...", "green", attrs=["bold"]))
    print("=" * 60)

    # Registrar temps d'inici
    temps_inici = time.time()

    try:
        # Entrenar
        if resume_from:
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            trainer.train()

        temps_total = time.time() - temps_inici

        print(colored("\n✅ ENTRENAMENT COMPLETAT!", "green", attrs=["bold"]))

        # Guardar model final
        print(colored("\n💾 Guardant model final...", "blue"))
        trainer.save_model()
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        # Guardar informació del format utilitzat
        formato_info = {
            "especialitat": ESPECIALITAT,
            "formato_original": formato_detectado,
            "total_ejemplos": len(dataset),
            "fecha_entrenamiento": datetime.now().isoformat(),
        }
        with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
            json.dump(formato_info, f, indent=2)

        # Anàlisi de resultats
        analitzar_resultats(trainer, OUTPUT_DIR, temps_total)

        # Test ràpid
        test_rapid_model(model, tokenizer, ESPECIALITAT)

        # Instruccions finals
        print(colored("\n📋 SEGÜENTS PASSOS:", "cyan", attrs=["bold"]))
        print("=" * 60)
        print(colored("1. Revisar gràfics:", "yellow"))
        print(f"   • {OUTPUT_DIR}/training_analysis.png")
        print(colored("2. Validar checkpoint:", "yellow"))
        print(f"   • python validar_checkpoint.py {OUTPUT_DIR}/checkpoint-XXX")
        print(colored("3. Si cal més entrenament:", "yellow"))
        print(f"   • python trainer.py {ESPECIALITAT}  (continuarà automàticament)")
        print(colored("4. Fer merge quan estigui llest:", "yellow"))
        print(f"   • python merge_lora.py {OUTPUT_DIR}")

        print(
            colored(
                f"\n🎉 Model {ESPECIALITAT} entrenat amb èxit! (Format: {formato_detectado})",
                "green",
                attrs=["bold"],
            )
        )

    except KeyboardInterrupt:
        temps_total = time.time() - temps_inici
        print(colored("\n⚠️ Entrenament interromput per l'usuari!", "yellow"))
        print(colored("💾 Guardant checkpoint d'emergència...", "blue"))

        emergency_dir = OUTPUT_DIR / "emergency_checkpoint"
        trainer.save_model(str(emergency_dir))
        tokenizer.save_pretrained(str(emergency_dir))

        print(colored(f"✅ Checkpoint guardat a: {emergency_dir}", "green"))
        analitzar_resultats(trainer, OUTPUT_DIR, temps_total)

    except Exception as e:
        print(colored(f"\n❌ ERROR: {str(e)}", "red"))
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
