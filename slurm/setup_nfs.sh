#!/bin/bash
# =====================================================================
# recsys-fairness — pré-instalação das dependências no NFS
#
# RODE ISTO NO NÓ DE CONTROLE (sci-slurm-server), que tem internet boa.
# Os nós de GPU têm rede instável; não dá para contar com pip/uv no meio
# do job (ver BOAS-PRATICAS.md, item 7). A estratégia:
#
#   * a imagem do container (pytorch25-cuda128.sif) já traz PyTorch com
#     CUDA 12.8 — necessário para as RTX 5090 (Blackwell);
#   * as libs do projeto são instaladas no NFS (--pylibs), usando o Python
#     do container. O RecBole 1.2.1 é instalado sem suas dependências
#     automáticas porque exige Ray <=2.6.3, indisponível para Python 3.12;
#   * as dependências usadas no treino são instaladas separadamente, sem
#     substituir o PyTorch/CUDA fornecido pela imagem.
#
# O job (run-array.sbatch) injeta essa pasta via PYTHONPATH=$RF_PYLIBS,
# depois do diretório do projeto, e NUNCA reinstala torch.
#
# Uso (no servidor):
#   ./slurm/setup_nfs.sh                     # usa os caminhos padrão
#   ./slurm/setup_nfs.sh --pylibs /caminho   # destino alternativo
#   ./slurm/setup_nfs.sh --image /img.sif    # imagem alternativa
#
# Reexecutar é seguro: o pip atualiza os pacotes no destino.
# =====================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

if (( EUID == 0 )); then
    echo "ERRO: rode este script sem sudo, com seu usuário do cluster." >&2
    exit 1
fi

CLUSTER_USER="$(id -un)"
IMAGE="${RF_CONTAINER:-/mnt/cluster-slurm/images/pytorch25-cuda128.sif}"
PYLIBS="${RF_PYLIBS:-/mnt/cluster-nfs/datasets/$CLUSTER_USER/pylibs}"
INDEX_URL="https://pypi.org/simple"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pylibs)    PYLIBS="$2"; shift 2 ;;
        --image)     IMAGE="$2"; shift 2 ;;
        --index-url) INDEX_URL="$2"; shift 2 ;;
        -h|--help)   sed -n '/^# recsys-fairness/,/^# Reexecutar/p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "opção desconhecida: $1" >&2; exit 2 ;;
    esac
done

[[ -e "$IMAGE" ]] || { echo "ERRO: imagem não encontrada: $IMAGE" >&2; exit 1; }

echo "recsys-fairness — pré-instalação de libs no NFS"
echo "  projeto : $PROJECT_ROOT"
echo "  imagem  : $IMAGE"
echo "  destino : $PYLIBS"
echo

mkdir -p "$PYLIBS"
[[ -w "$PYLIBS" ]] || { echo "ERRO: sem permissão de escrita em: $PYLIBS" >&2; exit 1; }

# RecBole declara Ray e Torch como dependências obrigatórias, mas o treino
# deste projeto não usa Ray e o Torch correto já vem da imagem CUDA 12.8.
# Fixamos as bibliotecas numéricas para evitar versões incompatíveis com
# RecBole 1.2.1. ipykernel é usado só nos notebooks locais.
PACKAGES=(
    "numpy==1.26.4"
    "scipy==1.15.3"
    "pandas==2.3.3"
    "scikit-learn==1.7.2"
    "matplotlib==3.10.9"
    "tqdm>=4.48.2"
    "colorlog==4.7.2"
    "colorama==0.4.4"
    "PyYAML>=5.1.0"
    "tensorboard>=2.5.0"
    "tabulate>=0.8.10"
    "plotly>=4.0.0"
    "texttable>=0.9.0"
    "psutil>=5.9.0"
)

echo ">> instalando dependências do treino no destino..."
apptainer exec \
    -B /mnt/cluster-nfs:/mnt/cluster-nfs \
    -B /mnt/cluster-slurm:/mnt/cluster-slurm \
    "$IMAGE" \
    python -m pip install \
        --index-url "$INDEX_URL" \
        --target="$PYLIBS" \
        --upgrade \
        "${PACKAGES[@]}"

echo
echo ">> instalando RecBole e thop sem baixar Ray ou outro PyTorch..."
apptainer exec \
    -B /mnt/cluster-nfs:/mnt/cluster-nfs \
    -B /mnt/cluster-slurm:/mnt/cluster-slurm \
    "$IMAGE" \
    python -m pip install \
        --index-url "$INDEX_URL" \
        --target="$PYLIBS" \
        --upgrade \
        --no-deps \
        "recbole==1.2.1" \
        "thop==0.1.1.post2209072238"

echo
echo ">> removendo qualquer stack antiga de torch/nvidia/triton do destino..."
# Pacotes deixados por uma instalação anterior não podem sombrear os da imagem.
for pattern in \
    "torch" "torch-*" "torchgen" "functorch" \
    "nvidia_*" "nvidia" \
    "triton" "triton-*" \
    "sympy" "sympy-*" \
    ; do
    find "$PYLIBS" -mindepth 1 -maxdepth 1 -iname "$pattern" -exec rm -rf -- {} +
done

echo
echo ">> verificação: imports do treino usando o torch do container + libs do NFS"
apptainer exec \
    -B /mnt/cluster-nfs:/mnt/cluster-nfs \
    -B /mnt/cluster-slurm:/mnt/cluster-slurm \
    "$IMAGE" \
    env PYTHONPATH="$PROJECT_ROOT:$PYLIBS" RF_PYLIBS="$PYLIBS" \
    python - <<'PY'
import os
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import sklearn
import torch
import recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from src.evaluators.model_evaluator_interface import IModelEvaluator
from src.scripts.evaluation import analyze_fairness

pylibs = Path(os.environ['RF_PYLIBS']).resolve()
torch_path = Path(torch.__file__).resolve()
if torch_path.is_relative_to(pylibs):
    raise RuntimeError(f'PyTorch veio do NFS, não do container: {torch_path}')
if torch.version.cuda != '12.8':
    raise RuntimeError(f'CUDA do PyTorch inesperado: {torch.version.cuda}')
if numpy.__version__ != '1.26.4':
    raise RuntimeError(f'NumPy inesperado: {numpy.__version__}')
if recbole.__version__ != '1.2.1':
    raise RuntimeError(f'RecBole inesperado: {recbole.__version__}')

print('torch       :', torch.__version__, '| CUDA', torch.version.cuda, '|', torch_path)
print('numpy       :', numpy.__version__)
print('scipy       :', scipy.__version__)
print('pandas      :', pandas.__version__)
print('scikit-learn:', sklearn.__version__)
print('matplotlib  :', matplotlib.__version__)
print('recbole     :', recbole.__version__)
print('OK: imports do treino disponíveis com torch do container')
PY

echo
echo "Pronto. As libs estão em: $PYLIBS"
echo "O job usa PYTHONPATH=<projeto>:$PYLIBS e o torch SEMPRE vem do container."
