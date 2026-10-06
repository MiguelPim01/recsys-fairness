# recsys-fairness no cluster Slurm — desenho e conceitos

Este documento explica **como** o projeto roda no cluster de GPUs do
departamento e **por quê** foi desenhado assim. Para o dia a dia ("quero
fazer X → rode Y"), veja `OPERACAO.md`. Para o cluster em si (acesso,
máquinas, partições), veja o material do boilerplate (`GUIA-SLURM.md`,
`BOAS-PRATICAS.md`).

## A ideia em uma frase

A campanha é um conjunto de **seeds**. O ETL é executado manualmente antes
da campanha; cada seed executa apenas
`sample → prepare → train/eval → fairness`. As seeds
são distribuídas entre as máquinas reservadas, cada máquina roda uma fatia,
e tudo é **retomável**: um job morto a qualquer momento é resubmetido e
continua de onde parou, sem refazer o que já ficou pronto. Cada worker
confere os dados processados antes de iniciar uma seed pendente.

## Por que isto cumpre o "contrato" do template avançado

O cluster é compartilhado e imprevisível (a partição `gpu` mata jobs em 1
dia; reservas acabam; máquinas caem). O template avançado só funciona bem se
o projeto tiver três propriedades (ver `BOAS-PRATICAS.md`). Aqui elas estão
assim:

1. **Checkpoint + retomada.** A unidade de retomada é o par
   *modelo × dataset* dentro de uma seed. Quando um treino termina, seu
   checkpoint `.pth` é registrado no `experiment.json` (com SHA-256). Esse
   registro — gravado **por último** — é o marcador de "concluído". Um job
   morto no meio deixa no máximo um `.pth` órfão (não registrado), que é
   descartado e refeito na próxima execução. O RecBole não retoma uma época
   no meio, mas essa granularidade (um treino por vez) já torna a perda
   máxima pequena.

2. **Idempotência.** Rodar de novo é barato e seguro:
   - seeds já `complete` são **puladas**;
   - dentro de uma seed parcial, a amostra e os splits são **reusados**
     (manifestos conferem hash), e os modelos já registrados são **pulados**
     pelo avaliador;
   - só o que falta realmente roda.

3. **"Quanto falta?"** `src.utils.campaign pending` imprime um único número:
   quantas seeds deste worker ainda não estão `complete`. É o que o
   encadeamento usa para decidir se vale resubmeter.

## Divisão do trabalho: por índice, não multi-GPU

Cada máquina do cluster tem **1 GPU**. Então não há DDP/multi-GPU: as seeds
são divididas por índice. Com `worker_count` máquinas, o worker `K` pega as
seeds cuja posição satisfaz `posição % worker_count == K`. É determinístico,
disjunto, e cada worker escreve só nas suas pastas de experimento — sem
coordenação entre eles. Se um worker cai, os outros nem percebem.

> Importante: ao mudar a lista de máquinas, **resubmeta todas juntas** para
> o `worker_count` ficar coerente (senão dois workers podem pegar a mesma
> seed). Ver `OPERACAO.md`.

## 1 nó reservado = 1 worker = 1 job que se auto-resubmete

O `submit.sh` manda **um job por nó reservado**, preso ao nó com `-w` e com
`--deadline` igual ao FIM da reserva daquele nó (lido de `reservations.txt`).
Cada job é um worker. Ao terminar — por concluir suas seeds **ou** por bater
no deadline — o `run-array.sbatch`:

- se foi **cancelado** (`scancel`, SIGTERM): **não** reencadeia;
- se morreu por **tempo** (SIGUSR1, 120s antes do limite) ou saiu com seeds
  pendentes: verifica `pending > 0` **e** se a reserva ainda vale com a
  margem; se ambos, **resubmete a si mesmo** com um novo deadline.
- **anti-loop:** jobs que duram menos de 5 min não reencadeiam (evita
  queimar a reserva repetindo um erro de configuração).

Por que um job por nó e não um array `0-(N-1)`: cada máquina pode ter um FIM
de reserva diferente, e um array Slurm aceita só um `--deadline` comum.

## Dependências e GPU: o container + libs no NFS

O software roda dentro do container `pytorch25-cuda128.sif`, que traz
**PyTorch com CUDA 12.8** (as RTX 5090 Blackwell exigem 12.8; versões
antigas de CUDA nem inicializam nelas). A imagem **não** traz as libs do
projeto (recbole, numpy<2, scikit-learn, matplotlib). Como os nós de GPU têm
rede instável, essas libs são **pré-instaladas uma vez no NFS** pelo
`setup_nfs.sh` (rodado no servidor) e entram no job via
`PYTHONPATH=<projeto>:<pylibs>`.

Detalhe crítico: o `setup_nfs.sh` **remove torch/nvidia/triton** da pasta de
libs do NFS depois de instalar o recbole, para que o torch bom (CUDA 12.8)
do container nunca seja sombreado por um torch sem CUDA arrastado como
dependência.

## Caminhos (convenção do cluster)

| O quê | Onde |
|---|---|
| Código | `/mnt/cluster-nfs/datasets/$USER/recsys-fairness` |
| Dados brutos | `data/raw/{lastfm_360k,yelp}` no computador local (não são necessários nos workers) |
| Dados processados | `.../recsys-fairness/data/processed/{lastfm,yelp}` (copiados após ETL local) |
| Libs no NFS | `/mnt/cluster-nfs/datasets/$USER/pylibs` |
| Imagem | `/mnt/cluster-slurm/images/pytorch25-cuda128.sif` |
| Resultados | `.../recsys-fairness/results/<u>_<i>/seed_<k>/` |
| Reservas | `slurm/reservations.txt` |

Os resultados caem em `results/` **dentro do projeto**, que já está no NFS —
portanto compartilhado entre as máquinas. Opcionalmente, `--output` cria um
symlink de `results/` para uma pasta em `/mnt/cluster-nfs/resultados/$USER/`.

## As peças

```
slurm/
├── reservations.txt   você mantém = espelho da planilha do cluster
├── reservations.sh    (não mexa) responde "reservado? até quando?"
├── setup_nfs.sh       roda no servidor: pré-instala libs no NFS
├── submit.sh          você roda: 1 job por nó reservado, com deadline
├── run-array.sbatch   (não mexa) o worker; ao morrer por tempo, reencadeia
├── SLURM.md           este documento (desenho)
└── OPERACAO.md        receituário do dia a dia

src/utils/campaign.py  orquestra as seeds por worker (run / pending / status)
```

O `campaign.py` não reimplementa o pipeline: ele chama os mesmos
`scripts/*.sh` do projeto, uma seed por vez, pulando o que já está pronto.
Ele não transforma dados brutos; o ETL local e a cópia para o NFS são
pré-requisitos para iniciar novas seeds.
Assim há uma única fonte de verdade de como um experimento é produzido.
