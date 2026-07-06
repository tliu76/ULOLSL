"""
Part 3: Regularization study with STANDARD ADAM (best Part-2 HPs, kept fixed).

Do NOT switch to AdamW or retune lr/betas here — that would confound optimizer
effects with regularization. Only add/remove/alter regularization mechanisms.

Required techniques: L2 weight decay (coupled), early stopping, dropout,
label smoothing, and modest train-only input noise/augmentation.

Required comparisons: Adam baseline (no added reg), best SINGLE regularizer,
best COMBINATION. Selection uses VALIDATION evidence only; test is final-only.

Required analysis: val loss + test metrics under a fixed budget; whether reg
improved Macro-F1 / balanced acc / minority-class recall / val loss / stability;
whether it reduced overfitting (train-val gap); whether any class was harmed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import get_split, make_loaders
from .model import build_model
from .plotting import save_json, save_table
from .seeds import set_all_seeds
from .train import final_test_metrics, make_criterion, train_model

# Best Part-2 Adam hyperparameters — set these from Part 2 results and freeze.
ADAM_LR = 1e-3
ADAM_BETAS = (0.9, 0.999)


def _train_one(reg: dict, seed: int, split, epochs: int = C.EPOCHS):
    """Train standard Adam with one regularization config. `reg` keys:
    weight_decay, dropout, label_smoothing, input_noise, early_stop (bool)."""
    set_all_seeds(seed)
    train_loader, val_loader, test_loader = make_loaders(split, seed=seed)
    model = build_model(dropout_p=reg.get("dropout", 0.0))
    criterion = make_criterion(label_smoothing=reg.get("label_smoothing", 0.0))
    opt = torch.optim.Adam(
        [p for p in model.parameters()], lr=ADAM_LR, betas=ADAM_BETAS,
        weight_decay=reg.get("weight_decay", 0.0),   # coupled L2 (standard Adam)
    )
    log = train_model(
        model, train_loader, val_loader, opt, criterion, epochs=epochs, seed=seed,
        input_noise_std=reg.get("input_noise", 0.0),
        early_stop_patience=C.EARLY_STOP_PATIENCE if reg.get("early_stop") else None,
    )
    test_m = final_test_metrics(model, test_loader, criterion)
    gap = (log.val_loss[-1] - log.train_loss[-1]) if log.val_loss else None
    return log, test_m, gap


def run_part3(seeds=None) -> pd.DataFrame:
    """Baseline + single-regularizer sweep + a candidate best combination."""
    seeds = seeds or C.SEEDS
    split = get_split()

    # Configs to evaluate. Expand the single-regularizer sweeps as needed; keep
    # detailed grids in REPRO and report compact selection evidence in the paper.
    configs = {"adam_baseline": {}}
    for wd in C.WEIGHT_DECAY_GRID:
        configs[f"l2_{wd:g}"] = {"weight_decay": wd}
    for dp in C.DROPOUT_GRID:
        configs[f"dropout_{dp:g}"] = {"dropout": dp}
    for ls in C.LABEL_SMOOTHING_GRID:
        configs[f"labelsmooth_{ls:g}"] = {"label_smoothing": ls}
    for ns in C.INPUT_NOISE_GRID:
        configs[f"inputnoise_{ns:g}"] = {"input_noise": ns}
    configs["early_stop"] = {"early_stop": True}
    # TODO: set best combination from single-regularizer validation results.
    configs["best_combo"] = {"weight_decay": 1e-4, "dropout": 0.1, "label_smoothing": 0.05}

    rows = []
    for name, reg in configs.items():
        for seed in seeds:
            log, test_m, gap = _train_one(reg, seed, split)
            row = {"config": name, "seed": seed, "gen_gap": gap,
                   "grad_evals": log.total_grad_evals, "status": log.status,
                   "val_loss_final": log.val_loss[-1] if log.val_loss else None}
            row.update(test_m.summary_row())
            rows.append(row)
            save_json(log.to_dict(), f"part3_{name}_seed{seed}.json")

    df = pd.DataFrame(rows)
    save_table(df, "part3_regularization_results.csv")
    return df


if __name__ == "__main__":
    print(run_part3())
