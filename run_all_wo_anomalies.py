import os
import subprocess
from multiprocessing import Process

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PERCENTAGES = [70, 10, 1]
DATASET = "brazil"
PRETRAIN = "off"
LOG_DIR = os.path.join(WORK_DIR, "logs")


def _run(script, model, pct, gpu, label):
    log_path = os.path.join(LOG_DIR, f"{label}_{model}_{DATASET}_{pct}pct.log")
    print(f"[GPU {gpu}] START {label} {model} {pct}% → {log_path}")

    cmd = [
        "python", script,
        "--model_name", model,
        "--dataset", DATASET,
        "--train_percent", str(pct),
        "--pretrain", PRETRAIN,
        "--gpu", str(gpu),
    ]

    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, cwd=WORK_DIR, stdout=log_file, stderr=log_file)

    if result.returncode != 0:
        print(f"[GPU {gpu}] ERRO: {label} {model} {pct}% (exit {result.returncode}) — ver {log_path}")
    else:
        print(f"[GPU {gpu}] OK: {label} {model} {pct}%")


def run_model(model, gpu):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.chdir(WORK_DIR)

    for pct in PERCENTAGES:
        _run("finetuning_without_anomalies.py", model, pct, gpu, label="wo_anomalies")
        _run("finetuning.py",                  model, pct, gpu, label="with_anomalies")


if __name__ == "__main__":
    p0 = Process(target=run_model, args=("BERTPP", 0))
    p1 = Process(target=run_model, args=("MAMBA",  1))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    print("\nAmbos terminaram.")
