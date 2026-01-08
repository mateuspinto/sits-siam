import subprocess
import sys

DATASETS = ["brazil", "texas", "california"]
PERCENTAGES = [70, 10, 1, 0.1]


def run_shallow_models():
    total = len(DATASETS) * len(PERCENTAGES)
    current = 0

    for dataset in DATASETS:
        for percentage in PERCENTAGES:
            current += 1
            print(f"\n{'=' * 80}")
            print(
                f"Running [{current}/{total}]: dataset={dataset}, train_percent={percentage}"
            )
            print(f"{'=' * 80}\n")

            cmd = [
                sys.executable,
                "shallows.py",
                "--dataset",
                dataset,
                "--train_percent",
                str(percentage),
                "--n_trials",
                100,
                "--n_jobs",
                20,
            ]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    cwd="/home/m/git/sits-siam",
                    capture_output=False,
                )
                print(f"\n✓ Completed: dataset={dataset}, train_percent={percentage}")
            except subprocess.CalledProcessError as e:
                print(
                    f"\n✗ Error executing: dataset={dataset}, train_percent={percentage}"
                )
                print(f"Error code: {e.returncode}")

                continue
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f"Process completed! Total executions: {total}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    run_shallow_models()
