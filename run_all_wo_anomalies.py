import os
import subprocess
from multiprocessing import Process

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PERCENTAGES = [70, 10, 1]
DATASET = "brazil"
PRETRAIN = "MoCo"


def run_model(model, gpu):
    os.chdir(WORK_DIR)
    for pct in PERCENTAGES:
        print(f"\n[GPU {gpu}] START {model} | {DATASET} {pct}% | pretrain={PRETRAIN}")

        cmd = [
            "python", "finetuning_without_anomalies.py",
            "--model_name", model,
            "--dataset", DATASET,
            "--train_percent", str(pct),
            "--pretrain", PRETRAIN,
            "--gpu", str(gpu),
        ]

        result = subprocess.run(cmd, cwd=WORK_DIR)

        if result.returncode != 0:
            print(f"[GPU {gpu}] ERRO: {model} {DATASET} {pct}% (exit {result.returncode})")
        else:
            print(f"[GPU {gpu}] OK: {model} {DATASET} {pct}%")


if __name__ == "__main__":
    p0 = Process(target=run_model, args=("BERTPP", 0))
    p1 = Process(target=run_model, args=("MAMBA",  1))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    print("\nAmbos terminaram.")
