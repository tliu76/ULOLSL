# OL Report — Optimization & Uncertainty in Learning (CS7641, Summer 2026)

A controlled optimization study on the Forest **Covertype** dataset using the
**fixed PyTorch backbone from the SL Report**. The goal is to explain *how*
training interventions (randomized optimization, Adam-family ablations,
regularization) affect validation loss, Macro-F1, balanced accuracy, class-level
behavior, stability, and compute cost — not just to maximize accuracy.

## Layout
```
ol-report/
├── requirements.txt
├── src/
│   ├── config.py          # ALL fixed settings (paths, seeds, budgets, backbone)
│   ├── seeds.py           # seed control (>=3 seeds for main claims)
│   ├── data.py            # SL sample reuse, stratified split, train-only scaling
│   ├── model.py           # PyTorch MLP backbone + freeze/flatten utils for RO
│   ├── metrics.py         # acc, macro-F1, balanced acc, confusion, per-class
│   ├── train.py           # shared loop + compute accounting (grad evals, time)
│   ├── adam_no_bias.py    # custom "Adam w/o bias correction" (Part 2)
│   ├── ro_algorithms.py   # RHC / SA / GA on flat param vector (func-eval budget)
│   ├── plotting.py        # figures + results IO
│   ├── baseline.py        # SL SGD-only baseline (OL reference point)
│   ├── part1_ro.py        # Part 1: RO on final 1-3 layers
│   ├── part2_adam.py      # Part 2: optimizer ablations + sensitivity heatmaps
│   ├── part3_reg.py       # Part 3: regularization study (standard Adam)
│   ├── part4_integrated.py# Part 4 (extra credit): integrated best combination
│   └── repro.py           # REPRO environment/config capture
├── experiments/run_all.py # CLI: baseline | part1 | part2 | part3 | part4 | all
└── outputs/{figures,results,logs}/
```

## Before running — required TODOs
1. **Backbone** (`config.HIDDEN_SIZES`, `model.CovertypeMLP`): set to your
   *finalized SL architecture* so OL is comparable to SL.
2. **Sample continuity** (`config.SAMPLE_SIZE`, `SAMPLE_SEED`, `SL_SAMPLE_PATH`):
   point at your SL ~20k stratified sample. `data.py` warns if it rebuilds one.
3. **`ELL_VAL_LOSS`**: pick the speed-to-target threshold from a train/val pilot
   (never from test).
4. **Best HPs**: after Part 2, set `part3_reg.ADAM_LR/ADAM_BETAS`; after Parts
   1–3, set winners in `part4_integrated.py`.

## Run
```bash
pip install -r ol-report/requirements.txt
cd ol-report
python -m experiments.run_all baseline   # confirm SL baseline first
python -m experiments.run_all part2       # then ablations, etc.
python -m src.repro                       # capture env for REPRO sheet
```

## Compute-accounting rules (built in)
- gradient-based: 1 minibatch backward = 1 gradient evaluation/update (`train.py`)
- RO: 1 validation-loss objective call = 1 function evaluation (`ro_algorithms.py`)
- wall-clock recorded for every run; compare GA by function evals, not generations
- test set touched only at final evaluation; scaler fit on train only

## Deliverables (do not forget)
- `OL_Report_{GTusername}.pdf` — IEEE `\documentclass[conference]{IEEEtran}`,
  8 pages max, >= 2 peer-reviewed sources cited inline (IEEE style), AI Use
  Statement, hypothesis stated early and resolved with quantitative evidence.
- `REPRO_OL_{GTusername}.pdf` — read-only Overleaf link, single GT Enterprise
  GitHub commit hash, run commands, env/versions, seeds, full heatmaps/seed
  tables, detailed sweeps (`src/repro.py` output helps here).
```
```
