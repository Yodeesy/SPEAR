# args.py
import argparse
import os

parser = argparse.ArgumentParser(description="SPEAR: Training-Free Federated Graph Learning")

# ==============================================================================
# 1. Environment & Paths
# ==============================================================================
current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')

if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)

log_path = os.path.join(os.path.dirname(current_path), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path, exist_ok=True)

env_group = parser.add_argument_group('Environment')
env_group.add_argument("--dataset", type=str, default="cora", help="Dataset name: cora, texas, actor, chameleon, etc.")
env_group.add_argument("--dataset_dir", type=str, default=root_dir, help="Root path for datasets")
env_group.add_argument("--logs_dir", type=str, default=log_path, help="Path to save logs")
env_group.add_argument("--device_id", type=int, default=0, help="GPU device ID")
env_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

# ==============================================================================
# 2. Data Partitioning (Non-IID Settings)
# ==============================================================================
data_group = parser.add_argument_group('Data Partitioning')
data_group.add_argument("--task", type=str, default="node_classification")
data_group.add_argument("--num_clients", type=int, default=10, help="Number of clients (K)")
data_group.add_argument("--dirichlet_beta", type=float, default=0.05, help="Dirichlet Beta (smaller = more Non-IID)")
data_group.add_argument("--least_samples", type=int, default=10, help="Min samples per client")
data_group.add_argument("--dataset_split_metric", type=str, default="transductive", choices=["transductive"])
data_group.add_argument("--train_val_test_split", type=float, nargs='+', default=[0.2, 0.4, 0.4])
data_group.add_argument("--dirichlet_try_cnt", type=int, default=10000)

# ==============================================================================
# 3. Federated Learning Settings
# ==============================================================================
fl_group = parser.add_argument_group('Federated Learning')
fl_group.add_argument("--fed_algorithm", type=str, default="SPEAR", help="Federated algorithm (e.g., SPEAR, FedAvg)")
fl_group.add_argument("--num_rounds", type=int, default=1, help="Total communication rounds (Set to 1 for SPEAR)")
fl_group.add_argument("--cl_sample_rate", type=float, default=1.0, help="Client sampling rate")
fl_group.add_argument("--T_L", type=int, default=1, help="Local training epochs (Used by FedAvg, bypassed by SPEAR)")

# ==============================================================================
# 4. Global Model (Backbone GNN for Training-based Baselines)
# ==============================================================================
model_group = parser.add_argument_group('Backbone Model (For Baselines)')
model_group.add_argument("--model", type=str, default="GCN", help="GNN Backbone: GCN, SAGE, GAT")
model_group.add_argument("--hidden_dim", type=int, default=128, help="Hidden feature dimension")
model_group.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
model_group.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
model_group.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate for FedAvg")
model_group.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay (L2 penalty)")

# ==============================================================================
# 5. SPEAR Specific Settings (Training-Free)
# ==============================================================================
spear_group = parser.add_argument_group('SPEAR Settings')
spear_group.add_argument("--alpha", type=float, default=0.15, help="Teleport probability for PPR topological smoothing")
spear_group.add_argument("--T", type=int, default=2, help="Number of PPR propagation steps")
spear_group.add_argument("--gamma", type=float, default=0.5, help="Statistical shrinkage hyperparameter")

args = parser.parse_args()