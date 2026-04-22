import os
import multiprocessing

# Configurações
WORK_DIR = "/home/user/sits-siam"
DATASETS = ["texas"] #, "texas", "california"] 
PERCENTAGES = [0.1, 1, 10, 70]
DEEP_MODELS = ["LSTM"] #["BERT", "CNN", "MAMBA", "BERTPP", "LSTM"]
GPUS = [0, 1]
PRETRAINS = ["siam"]
PPRETRAINS = ["MoCo"] #, "PMSN", "reconstruct", "FastSiam"]

DATASETS.reverse()
DEEP_MODELS.reverse()
PPRETRAINS.reverse()
PERCENTAGES.reverse()

def run_deep_models():
    if os.path.exists(WORK_DIR):
        os.chdir(WORK_DIR)

    for dataset in DATASETS:
        for model in DEEP_MODELS:
            for percentage in PERCENTAGES:
                for pree in PPRETRAINS:
                    if dataset == "brazil" and percentage == 0.1:
                        continue
    
                    print(f"🚀 Rodando: {model} | {dataset} ({percentage}%) na GPU 0")
                    
                    cmd = (
                        f"python finetuning.py "
                        f"--model_name {model} "
                        f"--dataset {dataset} "
                        f"--train_percent {percentage} "
                        f"--gpu 0 "
                        f"--pretrain {pree}"
                    )
                    
                    exit_code = os.system(cmd)
                    
                    if exit_code != 0:
                        print(f"❌ Erro: {model} em {dataset}")
                    else:
                        print(f"✅ Sucesso: {model} em {dataset}")

def run_in_season():
    if os.path.exists(WORK_DIR):
        os.chdir(WORK_DIR)


    for pree in ["MoCo", "off"]:
        for ddays in range(30, 331, 30):
    
            print(f"🚀 Rodando in season: {ddays} dias | (pretreino = {pree}) na GPU 1")
            
            cmd = (
                f"python in_season.py "
                f"--num_days {ddays} "
                f"--pretrain {pree}"
            )
            
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                print(f"❌ Erro:  {ddays} dias | (pretreino = {pree}) ")
            else:
                print(f"✅ Sucesso:  {ddays} dias | (pretreino = {pree}) ")
    
def run_mlp():
    if os.path.exists(WORK_DIR):
        os.chdir(WORK_DIR)

    for dataset in DATASETS:
        for percentage in PERCENTAGES:
            if dataset == "brazil" and percentage == 0.1:
                continue

            print(f"🚀 Rodando: MLP | {dataset} ({percentage}%) na GPU 0")
            
            cmd = (
                f"python mlp.py "
                f"--dataset {dataset} "
                f"--train_percent {percentage} "
                f"--gpu 1 "
            )
            
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                print(f"❌ Erro: MLP em {dataset} - {percentage}")
            else:
                print(f"✅ Sucesso: MLP em {dataset} - {percentage}")
                    
def pretrain_deep_models():
    if os.path.exists(WORK_DIR):
        os.chdir(WORK_DIR)

    for dataset in DATASETS:
        for model in DEEP_MODELS:
            for pretrain in PRETRAINS:

                print(f"🚀 Rodando pretreino: {model} | {dataset} na GPU 0 | {dataset}")
                
                cmd = (
                    f"python pretrain_{pretrain}.py "
                    f"--model_name {model} "
                    f"--dataset {dataset} "
                    f"--gpu 0"
                )
                
                exit_code = os.system(cmd)
                
                if exit_code != 0:
                    print(f"❌ Erro: {model} em {dataset}")
                else:
                    print(f"✅ Sucesso: {model} em {dataset}")

def run_shallow_models():
    # Muda para o diretório de trabalho uma vez
    if os.path.exists(WORK_DIR):
        os.chdir(WORK_DIR)
    
    total = len(DATASETS) * len(PERCENTAGES)
    count = 0

    for dataset in DATASETS:
        for percentage in PERCENTAGES:

            if dataset=="brazil" and percentage==0.1:
                continue

            count += 1
            print(f"\n--- [{count}/{total}] Rodando: {dataset} ({percentage}%) ---")

            # Monta o comando como uma string simples
            cmd = (
                f"python shallows.py "
                f"--dataset {dataset} "
                f"--train_percent {percentage} "
                f"--n_trials 100 "
                f"--n_jobs 20"
            )

            # Executa usando o sistema operacional diretamente
            exit_code = os.system(cmd)

            if exit_code != 0:
                print(f"❌ Erro ao executar {dataset} com {percentage}% (Código: {exit_code})")
            else:
                print(f"✅ Sucesso: {dataset} {percentage}%")

if __name__ == "__main__":
    # run_mlp()
    # run_shallow_models()
    # run_deep_models()
    # pretrain_deep_models()
    run_in_season()