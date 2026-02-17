import sys
import subprocess
from pathlib import Path


def run_step(step_name: str, command: list[str]) -> bool:
    """EXECUTA UM PASSO DO PIPELINE

    Args:
        step_name (str): _description_
        command (list[str]): _description_

    Returns:
        bool: _description_
    """
    print(f"\n{'=' * 60}")
    print(f"EXECUTANDO: {step_name}")
    print(f"\n{'=' * 60}")

    result = subprocess.run(command, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Erro em: {step_name}")

    print(f"{step_name} CONCLUÍDO!!")
    return True


def check_requirements() -> bool:
    """VERIFICA SE OS ARQUIVOS NECESSÁRIOS EXISTEM"""
    required = [
        Path("data/raw/telco.csv"),
        Path("params.yaml"),
        Path("src/data/prepare.py"),
        Path("src/model/train.py"),
        Path("src/model/evaluation.py"),
    ]

    all_ok = True
    for path in required:
        if path.exists():
            print(f"{path}")
        else:
            print(f"{path} não encontrado")
            all_ok = False
    
    return all_ok


def main() -> None:
    """PIPELINE COMPLETO: prepare -> train -> evaluate -> API"""
    print("=" * 60)
    print("CHURN PREDICTION - PIPELINE COMPLETO")
    print("=" * 60)

    # 1. Verificar requisitos
    if not check_requirements():
        print(f"Requisitos não atendidos")
        sys.exit(1)

    # 2. Pipeline de steps
    steps = [
        ("Preprocessamento dos dados", ["python", "-m", "src.data.prepare"]),
        ("Treinamento do modelo", ["python", "-m", "src.model.train"]),
        ("Avaliação do modelo", ["python", "-m", "src.model.evaluation"]),
    ]

    # 3. Executar steps
    for step_name, command in steps:
        sucess = run_step(step_name, command)
        if not sucess:
            print(f"Pipeline interrompido em: {step_name}")
            sys.exit(1)

    # 4. Verificar artefatos gerados
    print(f'\n{"="*60}')
    print("ARTEFATOS GERADOS")
    print(f'\n{"="*60}')

    artifacts = [
        Path("data/processed/train.csv"),
        Path("data/processed/test.csv"),
        Path("models/model.keras"),
        Path("models/preprocessor.pkl"),
        Path("metrics/training.json"),
        Path("metrics/evaluation.json"),
    ]

    for artifact in artifacts:
        status = "OK" if artifact.exists() else "NÃO EXISTE"
        print(f"{artifact} {status}")

    # 5. Subir  API
    print(f"\n{'='*60}")
    print("INICIANDO API")
    print("=" * 60)
    print("Acesse: http://localhost:8000")
    print("Pressione CTRL+C para encerrar")
    print("=" * 60)

    subprocess.run(["python", "-m", "app.api"])


if __name__ == "__main__":
    main()
