#!/usr/bin/env python3
"""
MONITOR DEFINITIU - Llegeix del checkpoint MÉS RECENT
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from termcolor import colored


def monitor_definitiu(output_dir):
    """Monitor que sempre llegeix del checkpoint més recent"""
    output_path = Path(output_dir)

    print(colored("=" * 70, "cyan"))
    print(colored("   📊 MONITOR DEFINITIU ATHENA", "cyan", attrs=["bold"]))
    print(colored("=" * 70, "cyan"))
    print(colored("Monitoritzant checkpoint més recent...", "yellow"))

    ultim_checkpoint = None

    try:
        while True:
            # Buscar el checkpoint MÉS RECENT
            checkpoints = sorted(
                [d for d in output_path.glob("checkpoint-*") if d.is_dir()],
                key=lambda x: (
                    int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0
                ),
            )

            if checkpoints:
                checkpoint_actual = checkpoints[-1]  # El més recent

                # Si ha canviat el checkpoint, actualitzar
                if checkpoint_actual != ultim_checkpoint:
                    ultim_checkpoint = checkpoint_actual

                    # Llegir dades del checkpoint
                    state_file = checkpoint_actual / "trainer_state.json"

                    if state_file.exists():
                        try:
                            with open(state_file, "r") as f:
                                state = json.load(f)

                            print("\033[H\033[J", end="")  # Clear screen

                            print(colored("=" * 70, "cyan"))
                            print(
                                colored(
                                    "   📊 MONITOR DEFINITIU ATHENA",
                                    "cyan",
                                    attrs=["bold"],
                                )
                            )
                            print(colored("=" * 70, "cyan"))

                            # Informació bàsica
                            current_step = state.get("global_step", 0)
                            epoch = state.get("epoch", 0)

                            print(colored(f"📁 Model: {output_path.name}", "yellow"))
                            print(
                                colored(
                                    f"🎯 Checkpoint: {checkpoint_actual.name}", "green"
                                )
                            )
                            print(
                                colored(
                                    f"🕒 Hora: {datetime.now().strftime('%H:%M:%S')}",
                                    "white",
                                )
                            )
                            print(
                                colored(
                                    f"📈 Step: {current_step}", "cyan", attrs=["bold"]
                                )
                            )
                            print(colored(f"📚 Epoch: {epoch:.2f}", "cyan"))

                            # Últimes mètriques
                            log_history = state.get("log_history", [])
                            losses = [l for l in log_history if "loss" in l]

                            if losses:
                                ultim_loss = losses[-1]["loss"]
                                print(
                                    colored(
                                        f"📉 Loss actual: {ultim_loss:.4f}", "green"
                                    )
                                )

                                # Tendència
                                if len(losses) > 1:
                                    tendencia = losses[-1]["loss"] - losses[-2]["loss"]
                                    if tendencia < -0.01:
                                        print(
                                            colored(
                                                "📈 Tendència: ✅ Millorant!", "green"
                                            )
                                        )
                                    elif tendencia > 0.01:
                                        print(
                                            colored(
                                                "📈 Tendència: ⚠️  Empitjorant", "red"
                                            )
                                        )
                                    else:
                                        print(
                                            colored(
                                                "📈 Tendència: ➡️  Estable", "yellow"
                                            )
                                        )

                                # Progrés
                                if len(losses) > 10:
                                    millora_total = (
                                        (losses[0]["loss"] - losses[-1]["loss"])
                                        / losses[0]["loss"]
                                        * 100
                                    )
                                    print(
                                        colored(
                                            f"🏆 Millora total: {millora_total:.1f}%",
                                            "cyan",
                                        )
                                    )

                            # Recomanació
                            if losses:
                                loss_actual = losses[-1]["loss"]
                                if loss_actual < 0.5:
                                    print(
                                        colored(
                                            "\n💡 RECOMANACIÓ: 🎯 Excel·lent! Gairebé llest",
                                            "green",
                                        )
                                    )
                                elif loss_actual < 0.8:
                                    print(
                                        colored(
                                            "\n💡 RECOMANACIÓ: ✅ Va molt bé, continua",
                                            "green",
                                        )
                                    )
                                elif loss_actual < 1.2:
                                    print(
                                        colored(
                                            "\n💡 RECOMANACIÓ: ⚠️  Acceptable, necessita més temps",
                                            "yellow",
                                        )
                                    )
                                else:
                                    print(
                                        colored(
                                            "\n💡 RECOMANACIÓ: ❌ Necessita més entrenament",
                                            "red",
                                        )
                                    )

                            print(colored("\n" + "=" * 70, "cyan"))
                            print(
                                colored(
                                    "S'actualitza quan es crea nou checkpoint", "white"
                                )
                            )

                        except Exception as e:
                            print(f"❌ Error llegint: {e}")
                else:
                    # Mostrar punt d'activitat
                    print(".", end="", flush=True)
            else:
                print(colored("⏳ Esperant primer checkpoint...", "yellow"))

            time.sleep(10)  # Revisar cada 10 segons

    except KeyboardInterrupt:
        print(colored("\n\n✋ Monitor aturat", "yellow"))


def main():
    if len(sys.argv) < 2:
        output_dirs = list(Path("output").glob("lora_*"))

        if not output_dirs:
            print(colored("❌ No s'han trobat models", "red"))
            sys.exit(1)

        print(colored("📂 Models:", "cyan"))
        for i, d in enumerate(output_dirs):
            print(f"  {i+1}. {d.name}")

        choice = input(colored("Selecciona model (número): ", "yellow"))

        try:
            output_dir = output_dirs[int(choice) - 1]
        except:
            print(colored("❌ Selecció invàlida!", "red"))
            sys.exit(1)
    else:
        output_dir = Path(sys.argv[1])

    monitor_definitiu(output_dir)


if __name__ == "__main__":
    main()
