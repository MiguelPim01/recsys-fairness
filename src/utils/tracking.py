"""Small W&B boundary shared by training and fairness analysis."""

import hashlib
import os
from contextlib import contextmanager


def enabled() -> bool:
    if os.getenv("RECSYS_WANDB") != "1":
        return False
    if not os.getenv("WANDB_PROJECT"):
        raise ValueError("Set WANDB_PROJECT when RECSYS_WANDB=1")
    return True


def run_config(experiment_dir, dataset, model, user_limit, item_limit, seed, **extra):
    return {
        "experiment": experiment_dir.name,
        "dataset": dataset,
        "model": model.casefold(),
        "user_limit": user_limit,
        "item_limit": item_limit,
        "seed": seed,
        **extra,
    }


def record_recbole_config(config):
    if not enabled():
        return
    import wandb

    if wandb.run is None:
        return
    wandb.run.config.update({"recbole": {
        key: config[key] for key in (
            "learning_rate", "epochs", "eval_step", "stopping_step",
            "train_batch_size", "eval_batch_size", "valid_metric", "topk",
        )
    }})


@contextmanager
def track(name, job_type, config, *, stable=False):
    if not enabled():
        yield None
        return

    import wandb

    group = (
        f"u{config['user_limit']}-i{config['item_limit']}"
        f"-{config['experiment']}"
    )
    identity = (
        hashlib.sha256(f"{group}:{name}:{job_type}".encode()).hexdigest()[:8]
        if stable else None
    )
    with wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.getenv("WANDB_ENTITY") or None,
        group=group,
        name=name,
        job_type=job_type,
        config=config,
        id=identity,
        resume="allow" if stable else None,
        save_code=False,
    ) as run:
        yield run
