# Rethinking One-Shot Federated Graph Learning: Training-Free Statistical Estimation
[![arXiv](https://img.shields.io/badge/arXiv-2609.06154-b31b1b.svg)](https://arxiv.org/abs/2609.06154)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

One-shot federated graph learning generally aims to train Graph Neural Networks (GNNs) across clients with disconnected subgraphs in a single communication round. Existing methods predominantly design advanced optimization strategies under the premise that local GNN training is indispensable. However, empirical observations reveal that under extreme non-IID conditions, local GNN training suffers from severe cross-client representation misalignment, becoming a major source of error rather than a remedy. Motivated by this, we reformulate one-shot FGL as a statistical estimation problem.
We propose SPEAR (Statistical Prototype Estimation with Adaptive Reliability), a completely training-free framework that directly computes topology-smoothed class prototypes from local graphs in the original feature space. The server then aggregates these prototypes using a sample-size-adaptive shrinkage estimator that down-weights unreliable local estimates, producing robust global class prototypes.
Extensive experiments across seven benchmarks demonstrate that SPEAR consistently achieves state-of-the-art accuracy under extreme heterogeneity. Moreover, SPEAR delivers at least an order-of-magnitude speedup over all baselines, reaching several orders of magnitude against generative and distillation-based methods.
Our findings suggest that training-free statistical estimation, rather than local GNN optimization, provides the key to robust and efficient one-shot federated graph learning.

---

##  Highlights

- **Zero Training & Backpropagation:** Completely bypasses local gradient descent. No learning rates, optimizers, or loss functions to tune.
- **Strictly One-Shot:** Requires only **a single communication round** to achieve state-of-the-art representation calibration.
- **Robust Against Extreme Heterogeneity:** Employs closed-form statistical shrinkage (James-Stein estimation) to conquer severe Non-IID skew ($\beta=0.05$), where gradient-based methods collapse.
- **Orders of Magnitude Faster:** Runs and evaluates across benchmark datasets in **seconds**, not hours.

---

## Environment Setup

Create and activate the conda environment:

```bash
conda create -n spear python=3.9 -y
conda activate spear

# Install PyTorch with CUDA 11.8 support
conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyG and dependencies
pip install torch_geometric==2.6.1
pip install numpy scipy scikit-learn
```

---

## Quick Start

Run SPEAR on Cora with extreme non-IID partitioning ($\beta=0.05$):

```bash
python main.py --fed_algorithm SPEAR --dataset cora --dirichlet_beta 0.05 --num_clients 10
```

To evaluate on other benchmarks (e.g., `citeseer`, `pubmed`, `coauthor-cs`, `wikics`, `ogbn-arxiv`):

```bash
python main.py --fed_algorithm SPEAR --dataset citeseer
```

---

## Repository Structure

```txt
.
├── main.py                     # Entry point for running algorithms
├── args.py                     # Hyperparameter configuration
├── algorithm/
│   ├── SPEAR.py                # Proposed training-free statistical framework
│   └── Base.py                 # Server and client base classes
├── backbone/                   # GNN models (for reproducing training-based baselines)
├── datasets/                   # Graph dataset loaders & Dirichlet partitioning
├── utils/                      # Evaluation metrics & helper functions
└── logs/                       # Auto-generated experiment outputs
```

---

## Reproducing Paper Results

All benchmark configurations follow the default parameters in `args.py`:

* **Heterogeneity:** Dirichlet distribution parameter $\beta = 0.05$ (10 clients)
* **Seeds:** Evaluated over multiple runs (`42`, `0`, `1`, `2027`, `2077`)
* **Hyperparameters:** Teleport probability $\alpha = 0.15$, Propagation depth $T = 2$, Shrinkage factor $\gamma = 0.5$

*Note: Baseline training-based GNN architectures are provided under `backbone/` solely for benchmark reproducibility.*

---
##Citation
If you find this work or code useful in your research, please consider citing:
```txt
@misc{zheng2026rethinkingoneshotfederatedgraph,
      title={Rethinking One-Shot Federated Graph Learning: Training-Free Statistical Estimation}, 
      author={Shutong Zheng and Sijia Chen},
      year={2026},
      eprint={2609.06154},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.06154}, 
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

