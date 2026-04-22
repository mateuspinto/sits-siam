import os
import subprocess
from multiprocessing import Process, Queue

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORK_DIR, "logs")

DATASETS      = ["brazil", "texas", "california"]
MODELS        = ["BERT", "BERTPP", "LSTM", "CNN", "MAMBA"]
PRETRAINS     = ["off", "reconstruct", "MoCo", "PMSN", "FastSiam"]
TRAIN_PERCENTS = [70, 10, 1, 0.1]


def worker(gpu, queue):
    os.makedirs(LOG_DIR, exist_ok=True)

    while True:
        item = queue.get()
        if item is None:
            break

        dataset, model, pretrain, train_percent = item
        label = f"{dataset}_{model}_{pretrain}_{train_percent}"
        log_path = os.path.join(LOG_DIR, f"openset_{label}_gpu{gpu}.log")
        print(f"[GPU {gpu}] START dataset={dataset} model={model} pretrain={pretrain} train_percent={train_percent} → {log_path}")

        cmd = [
            "python", "finetuning_open_set.py",
            "--dataset", dataset,
            "--model_name", model,
            "--pretrain", pretrain,
            "--train_percent", str(train_percent),
            "--gpu", str(gpu),
        ]

        with open(log_path, "w") as log_file:
            result = subprocess.run(cmd, cwd=WORK_DIR, stdout=log_file, stderr=log_file)

        if result.returncode != 0:
            print(f"[GPU {gpu}] ERRO: {label} (exit {result.returncode}) — ver {log_path}")
        else:
            print(f"[GPU {gpu}] OK: {label}")


if __name__ == "__main__":
    combos = [
        (dataset, model, pretrain, train_percent)
        for dataset in DATASETS
        for model in MODELS
        for pretrain in PRETRAINS
        for train_percent in TRAIN_PERCENTS
    ]

    q0 = Queue()
    q1 = Queue()

    for i, combo in enumerate(combos):
        if i % 2 == 0:
            q0.put(combo)
        else:
            q1.put(combo)

    q0.put(None)
    q1.put(None)

    p0 = Process(target=worker, args=(0, q0))
    p1 = Process(target=worker, args=(1, q1))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    print("\nTodos os experimentos open-set finalizados.")
