import os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = ["MAMBA", "BERTPP"]
PERCENTAGES = [70, 10, 1]
DATASET = "brazil"
PRETRAIN = "MoCo"
GPU = 0

if __name__ == "__main__":
    os.chdir(WORK_DIR)

    total = len(MODELS) * len(PERCENTAGES)
    count = 0

    for model in MODELS:
        for pct in PERCENTAGES:
            count += 1
            print(f"\n[{count}/{total}] {model} | {DATASET} {pct}% | pretrain={PRETRAIN}")

            cmd = (
                f"python finetuning_without_anomalies.py "
                f"--model_name {model} "
                f"--dataset {DATASET} "
                f"--train_percent {pct} "
                f"--pretrain {PRETRAIN} "
                f"--gpu {GPU}"
            )

            exit_code = os.system(cmd)

            if exit_code != 0:
                print(f"ERRO: {model} {DATASET} {pct}% (exit {exit_code})")
            else:
                print(f"OK: {model} {DATASET} {pct}%")
