import os
import subprocess
from multiprocessing import Process

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PERCENTAGES = [70, 10, 1]
DATASET = "brazil"
PRETRAIN = "off"
LOG_DIR = os.path.join(WORK_DIR, "logs")


def run_model(model, gpu):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.chdir(WORK_DIR)

    for pct in PERCENTAGES:
        log_path = os.path.join(LOG_DIR, f"{model}_{DATASET}_{pct}pct.log")
        print(f"[GPU {gpu}] START {model} {pct}% → {log_path}")

        cmd = [
            "python", "finetuning_without_anomalies.py",
            "--model_name", model,
            "--dataset", DATASET,
            "--train_percent", str(pct),
            "--pretrain", PRETRAIN,
            "--gpu", str(gpu),
        ]

        with open(log_path, "w") as log_file:
            result = subprocess.run(cmd, cwd=WORK_DIR, stdout=log_file, stderr=log_file)

        if result.returncode != 0:
            print(f"[GPU {gpu}] ERRO: {model} {pct}% (exit {result.returncode}) — ver {log_path}")
        else:
            print(f"[GPU {gpu}] OK: {model} {pct}%")


if __name__ == "__main__":
    p0 = Process(target=run_model, args=("BERTPP", 0))
    p1 = Process(target=run_model, args=("MAMBA",  1))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    print("\nAmbos terminaram.")
