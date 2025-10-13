# Federated and Ordinal Learning for Wildfire Risk Forecasting

## Abstract
Wildfire services require timely risk indicators that can be trained without relaxing territorial data-governance rules, yet federated learning algorithms such as FedALA and FedMOON have scarcely been evaluated for wildfire prediction—and not at all for ordinal classification. We address this gap by studying daily wildfire occurrence forecasting for 94 French departments in an ordinal multi-class formulation, comparing centralized training against three federated learning (FL) algorithms—FedAvg, FedALA, and FedMOON—across four client-partitioning schemes while explicitly examining confidentiality implications. The centralized gated recurrent unit (GRU) optimized with the multi-class Earth mover's distance with Kendall weighting (MCEWK) loss establishes the performance benchmark, delivering a 0.24 global Intersection-over-Union (IoU) and 0.44 IoU on extreme fire classes. Among FL strategies, FedALA yields the most stable global IoU (0.23) and generalization (0.11) when clients are grouped seasonally, while FedMOON under the Mediterranean partition attains the strongest extreme-event IoU (0.50). Conversely, department-level federation lags markedly because sparsely populated departments inject noisy updates that impede convergence. These results clarify both the performance trade-offs and the privacy limits of federated wildfire prediction, highlighting where additional mitigation is needed to make department-scale confidentiality viable.

## Overview
This repository collects the core loss implementations used in the accompanying study on ordinal wildfire forecasting. The losses support training recurrent models and federated learning strategies that must preserve ordinal structure and handle class imbalance.

- `loss.py` implements ordinal-aware training criteria, including binomial cross-entropy for ordered targets, a foreground dice loss, and the combined MCE + WK objective used in the paper.
- `loss_utils.py` provides helper routines for constructing cost matrices and class weights that are reused across experiments.
- `models.py` contains wrapper for centralize and federated training

## Usage
The loss classes are written for PyTorch-based workflows. Import the desired loss from `loss.py`, instantiate it with the number of classes used in your ordinal prediction task, and integrate it into your training loop. The helper utilities in `loss_utils.py` simplify building weighting schemes that reflect ordinal distances between classes.

## Citation
If you make use of this codebase or the associated study, please cite the work referenced in the abstract above.
