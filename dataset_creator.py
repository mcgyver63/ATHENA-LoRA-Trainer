#!/usr/bin/env python3
import os
import json
import PyPDF2
import fitz  # PyMuPDF
from pathlib import Path
import random


def extract_text_from_pdf(pdf_path):
    """Extreu text d'un PDF utilitzant PyMuPDF (més robust)"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        total_pages = len(doc)

        for page_num, page in enumerate(doc, 1):
            text += page.get_text()
            if page_num % 10 == 0:
                print(f"   Processades {page_num}/{total_pages} pàgines...")

        print(f"✅ Usant PyMuPDF - {len(text)} caràcters")
        return text
    except Exception as e:
        print(f"⚠️  Error amb PyMuPDF: {e}")
        return extract_text_pypdf2(pdf_path)


def extract_text_pypdf2(pdf_path):
    """Mètode alternatiu amb PyPDF2"""
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        print(f"✅ Usant PyPDF2 - {len(text)} caràcters")
        return text
    except Exception as e:
        print(f"❌ Error amb PyPDF2: {e}")
        return ""


def chunk_text(text, chunk_size=500, overlap=100):
    """Divideix text en chunks amb overlap"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Intentar tallar en punt final
        if end < len(text):
            last_period = chunk.rfind(".")
            if last_period > chunk_size * 0.5:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1

        if len(chunk.strip()) > 50:  # Només chunks amb contingut
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


def create_dataset_from_pdfs(input_dir, output_file, specialty, language="ES"):
    """Crea dataset des de PDFs"""

    # Prompts segons idioma
    if language.upper() == "CA":
        prompts = [
            "Explica aquest contingut:",
            "Resume aquest text:",
            "Descriu el següent:",
            "Quina informació proporciona aquest text?",
            "Explica aquest concepte:",
        ]
    else:  # ES
        prompts = [
            "Explica el siguiente contenido:",
            "Resume este texto:",
            "Describe lo siguiente:",
            "¿Qué información proporciona este texto?",
            "Explica este concepto:",
        ]

    pdf_files = list(Path(input_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No s'han trobat PDFs a {input_dir}")
        return

    print(f"\n📂 Buscant PDFs a: {input_dir}")
    print(f"✅ Trobats {len(pdf_files)} PDFs")

    all_samples = []

    for pdf_file in pdf_files:
        print(f"\n📄 Processant: {pdf_file.name}")

        # Extreure text
        text = extract_text_from_pdf(str(pdf_file))

        if not text:
            print(f"⚠️  No s'ha pogut extreure text de {pdf_file.name}")
            continue

        print(f"✅ Text extret: {len(text)} caràcters")

        # Crear chunks
        chunks = chunk_text(text)
        print(f"✅ Chunking: {len(text)} chars → {len(chunks)} chunks")

        # Crear samples
        for chunk in chunks:
            sample = {
                "instruction": random.choice(prompts),
                "input": "",
                "output": chunk,
            }
            all_samples.append(sample)

        print(f"✅ {len(chunks)} samples creats de {pdf_file.name}")

    # Guardar dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dataset complet creat!")
    print(f"📊 Total de samples: {len(all_samples)}")
    print(f"💾 Guardat a: {output_file}")
    print(f"\nℹ️  Ara pots usar-lo amb:")
    print(f"   python super_trainer.py")


def main():
    # Directori base
    base_dir = Path(__file__).parent

    # NOVA PREGUNTA: Idioma
    print("\n🌍 Selecciona l'idioma del dataset:")
    print("   CA - Català")
    print("   ES - Castellà/Español")
    language = input("\nIdioma (CA/ES): ").strip().upper()

    if language not in ["CA", "ES"]:
        print("⚠️  Idioma no vàlid. Usant ES per defecte.")
        language = "ES"

    lang_name = "Català" if language == "CA" else "Castellà"
    print(f"✅ Idioma seleccionat: {lang_name}")

    # Demanar especialitat
    specialty = input("\nNom de l'especialitat (ex: fisica, quimica): ").strip()

    # Paths
    input_dir = base_dir / "input_pdfs"
    output_file = base_dir / "datasets" / f"dataset_{specialty}.json"

    # Crear dataset
    create_dataset_from_pdfs(
        input_dir=str(input_dir),
        output_file=str(output_file),
        specialty=specialty,
        language=language,
    )


if __name__ == "__main__":
    main()
