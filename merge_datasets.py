#!/usr/bin/env python3
"""
Fusiona dos o més fitxers dataset_*.json en un de sol.
"""
import json
from pathlib import Path
import sys

def load_dataset(filepath):
    """Carrega un dataset des d'un fitxer JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"⚠️  Advertència: {filepath} no conté una llista. Potser el format no és el correcte.")
        return data
    except FileNotFoundError:
        print(f"❌ Error: No s'ha trobat el fitxer {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: No s'ha pogut llegir el JSON de {filepath}: {e}")
        return None

def save_dataset(data, filepath):
    """Guarda un dataset a un fitxer JSON."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Dataset fusionat guardat a {filepath}")
    except Exception as e:
        print(f"❌ Error: No s'ha pogut guardar el fitxer {filepath}: {e}")

def main():
    if len(sys.argv) < 4:
        print("Ús: python merge_datasets.py <fitxer_sortida.json> <fitxer1.json> <fitxer2.json> [<fitxer3.json> ...]")
        print("Exemple: python merge_datasets.py dataset_combinat.json dataset_InstruCAT.json dataset_Matematiques.json")
        sys.exit(1)

    output_filename = sys.argv[1]
    input_filenames = sys.argv[2:]

    base_path = Path("/home/joancarles/lora_training")
    datasets_folder = base_path / "datasets"
    output_path = datasets_folder / output_filename

    if not datasets_folder.exists():
        print(f"❌ Error: No s'ha trobat la carpeta {datasets_folder}")
        sys.exit(1)

    merged_data = []
    total_samples = 0

    print("🔄 Iniciant la fusió de datasets...")
    for filename in input_filenames:
        file_path = datasets_folder / filename
        print(f"📖 Llegint {filename}...")
        data = load_dataset(file_path)
        if data is not None:
            original_len = len(data)
            # Filtrar per assegurar que tots els elements són diccionaris (opcional, per seguretat)
            filtered_data = [item for item in data if isinstance(item, dict)]
            filtered_len = len(filtered_data)
            if filtered_len != original_len:
                print(f"⚠️  S'han filtrat {original_len - filtered_len} elements no vàlids de {filename}")
            
            merged_data.extend(filtered_data)
            total_samples += filtered_len
            print(f"   ➕ Afegides {filtered_len} mostres de {filename}")
        else:
            print(f"❌ S'ha omès {filename} per error.")
    
    if not merged_data:
        print("❌ No s'ha pogut carregar cap dataset o tots estaven buits.")
        sys.exit(1)

    print(f"📊 Total de mostres combinades: {total_samples}")
    
    # Opcional: barrejar les dades per evitar seqüències repetides (pot ajudar en l'entrenament)
    # import random
    # random.shuffle(merged_data)
    # print("🔀 Dataset barrejat per a millor entrenament.")

    save_dataset(merged_data, output_path)
    print(f"🎉 Fusió completada! Dataset final: {len(merged_data)} mostres.")

if __name__ == "__main__":
    main()
