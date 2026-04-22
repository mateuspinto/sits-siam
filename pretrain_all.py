import os
import subprocess
from multiprocessing import Process, Queue

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORK_DIR, "logs")

DATASETS = ["brazil", "texas", "california"]
MODELS = ["BERT", "BERTPP", "LSTM", "CNN", "MAMBA"]
PRETRAINS = ["siam", "moco", "pmsn", "reconstruct"]


def worker(gpu, queue):
    os.makedirs(LOG_DIR, exist_ok=True)

    while True:
        item = queue.get()
        if item is None:
            break

        pretrain, model, dataset = item
        label = f"{pretrain}_{model}_{dataset}"
        log_path = os.path.join(LOG_DIR, f"pretrain_{label}_gpu{gpu}.log")
        print(f"[GPU {gpu}] START pretrain={pretrain} model={model} dataset={dataset} → {log_path}")

        cmd = [
            "python", f"pretrain_{pretrain}.py",
            "--model_name", model,
            "--dataset", dataset,
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
        (pretrain, model, dataset)
        for dataset in DATASETS
        for model in MODELS
        for pretrain in PRETRAINS
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

    print("\nTodos os pretreinos finalizados.")
