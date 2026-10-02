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
#   * as libs do projeto que NÃO estão na imagem (recbole, numpy<2,
#     scikit-learn, matplotlib e suas transitivas) são instaladas aqui,
#     uma vez, numa pasta do NFS (--pylibs), usando o python do container;
#   * em seguida REMOVEMOS do destino qualquer torch/nvidia/triton que o
#     pip tenha arrastado como dependência — senão, com essa pasta no
#     PYTHONPATH, um torch SEM CUDA 12.8 sombrearia o torch bom do
#     container e a GPU (sobretudo a 5090) pararia de inicializar.
#
# O job (run-array.sbatch) injeta essa pasta via PYTHONPATH=$RF_PYLIBS,
# depois do diretório do projeto, e NUNCA reinstala torch.
#
# Uso (no servidor):
#   ./slurm/setup_nfs.sh                     # usa os caminhos padrão
#   ./slurm/setup_nfs.sh --pylibs /caminho   # destino alternativo
#   ./slurm/setup_nfs.sh --image /img.sif    # imagem alternativa
#
# Reexecutar é seguro: o pip apenas completa o que falta no destino.
# =====================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

IMAGE="${RF_CONTAINER:-/mnt/cluster-slurm/images/pytorch25-cuda128.sif}"
PYLIBS="${RF_PYLIBS:-/mnt/cluster-nfs/datasets/$USER/pylibs}"
INDEX_URL="https://pypi.org/simple"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pylibs)    PYLIBS="$2"; shift 2 ;;
        --image)     IMAGE="$2"; shift 2 ;;
        --index-url) INDEX_URL="$2"; shift 2 ;;
        -h|--help)   sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
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

# Pacotes do projeto que faltam na imagem. Casam com o pyproject.toml:
#   numpy<2.0.0, recbole>=1.2.1, scikit-learn>=1.7.2, matplotlib>=3.10.0
# (ipykernel é só para notebooks; não é necessário no cluster.)
PACKAGES=(
    "numpy<2.0.0"
    "recbole>=1.2.1"
    "scikit-learn>=1.7.2"
    "matplotlib>=3.10.0"
)

echo ">> instalando no destino (torch será removido depois para não sombrear o do container)..."
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
echo ">> removendo torch/nvidia/triton do destino (ficam a cargo do container)..."
# Remove qualquer stack de CUDA/torch que tenha vindo como dependência.
# Mantemos SOMENTE as libs puras do projeto; o torch bom é o do .sif.
for pattern in \
    "torch" "torch-*" "torchgen" "functorch" \
    "nvidia_*" "nvidia" \
    "triton" "triton-*" \
    "sympy" "sympy-*" \
    ; do
    find "$PYLIBS" -maxdepth 1 -iname "$pattern" -exec rm -rf {} + 2>/dev/null || true
done

echo
echo ">> verificação: importa recbole usando o torch do container + libs do NFS"
apptainer exec --nv \
    -B /mnt/cluster-nfs:/mnt/cluster-nfs \
    -B /mnt/cluster-slurm:/mnt/cluster-slurm \
    "$IMAGE" \
    bash -c "
        set -euo pipefail
        export PYTHONPATH='$PROJECT_ROOT:$PYLIBS'
        python - <<'PY'
import torch, numpy, sklearn, matplotlib
print('torch       :', torch.__version__, '| CUDA', torch.version.cuda)
print('numpy       :', numpy.__version__)
print('scikit-learn:', sklearn.__version__)
print('matplotlib  :', matplotlib.__version__)
import recbole
print('recbole     :', recbole.__version__)
assert numpy.__version__.startswith('1.'), 'numpy precisa ser <2 para o recbole 1.2.1'
print('OK: recbole importa sobre o torch do container')
PY
    "

echo
echo "Pronto. As libs estão em: $PYLIBS"
echo "O job usa PYTHONPATH=<projeto>:$PYLIBS e o torch SEMPRE vem do container."
