import os

WORK_DIR = "/home/m/git/sits-siam"
DATASETS = ["brazil", "texas", "california"]
PERCENTAGES = [70, 10, 1, 0.1]


def run_shallow_models():
    if os.path.exists(WORK_DIR):
        os.chdir(WORK_DIR)

    total = len(DATASETS) * len(PERCENTAGES)
    count = 0

    for dataset in DATASETS:
        for percentage in PERCENTAGES:
            count += 1
            print(f"\n--- [{count}/{total}] Rodando: {dataset} ({percentage}%) ---")

            cmd = (
                f"python shallows.py "
                f"--dataset {dataset} "
                f"--train_percent {percentage} "
                f"--n_trials 100 "
                f"--n_jobs 20"
            )

            exit_code = os.system(cmd)

            if exit_code != 0:
                print(
                    f"❌ Erro ao executar {dataset} com {percentage}% (Código: {exit_code})"
                )
            else:
                print(f"✅ Sucesso: {dataset} {percentage}%")


if __name__ == "__main__":
    run_shallow_models()
