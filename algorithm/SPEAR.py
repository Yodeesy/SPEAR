# algorithm/SPEAR.py
import os
import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, add_self_loops
from sklearn.metrics import f1_score

from algorithm.Base import BaseServer, BaseClient


class SPEARClient(BaseClient):
    """Client for the SPEAR framework.

    Performs local topological smoothing using Personalized PageRank (PPR)
    and extracts class-wise mean embeddings (prototypes).
    """

    def __init__(self, args, model, data):
        super(SPEARClient, self).__init__(args, model, data)
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes = args.num_classes
        self.alpha = getattr(args, 'alpha', 0.15)
        self.T = getattr(args, 'T', 2)

    def train(self):
        """Training is bypassed as SPEAR is a training-free framework."""
        return 0.0

    def _ppr_smooth(self, x, edge_index, num_nodes=None):
        """Applies Personalized PageRank (PPR) smoothing to node features.

        Args:
            x: Node feature matrix.
            edge_index: Graph edge indices.
            num_nodes: Total number of nodes in the graph.

        Returns:
            Smoothed node feature matrix.
        """
        if num_nodes is None:
            num_nodes = x.size(0)

        x_norm = F.normalize(x, p=2, dim=1)

        if edge_index.size(1) == 0:
            return x_norm

        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop
        deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)

        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row]

        adj_sparse = torch.sparse_coo_tensor(
            edge_index_loop,
            norm,
            (num_nodes, num_nodes),
            device=self.device
        )

        h_k = x_norm.clone()
        for _ in range(self.T):
            h_k = (1 - self.alpha) * x_norm + self.alpha * torch.sparse.mm(adj_sparse, h_k)

        return h_k

    def get_stats(self):
        """Computes and returns local class prototypes and sample counts.

        Returns:
            dict: A mapping from class index to local statistics ('mu' and 'n_c').
        """
        x = self.data.x.to(self.device)
        edge_index = self.data.edge_index.to(self.device)

        # Obtain smoothed features
        h_k = self._ppr_smooth(x, edge_index)

        train_mask = self.data.train_mask
        labels = self.data.y

        stats = {}
        for c in range(self.num_classes):
            mask_c = train_mask & (labels == c)
            n_c = mask_c.sum().item()
            if n_c > 0:
                mu = h_k[mask_c].mean(dim=0).cpu()
                stats[c] = {
                    'mu': mu,
                    'n_c': n_c
                }
        return stats


class SPEARServer(BaseServer):
    """Server for the SPEAR framework.

    Aggregates local client prototypes using James-Stein statistical shrinkage
    and performs global evaluation.
    """

    def __init__(self, args, clients, model, data, logger):
        super(SPEARServer, self).__init__(args, clients, model, data, logger)
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes = args.num_classes
        self.gamma = getattr(args, 'gamma', 0.5)
        self.alpha = getattr(args, 'alpha', 0.15)
        self.T = getattr(args, 'T', 2)

        self.args = args
        self.global_prototypes = None

    def run(self):
        """Executes the federated aggregation and evaluation pipeline."""
        print(f"\n{'=' * 60}")
        print("Starting SPEAR Server Initialization")
        print(f"{'=' * 60}")

        metrics = self.aggregate()
        acc, f1 = self.global_evaluate()
        self.stop_training = True

        print(f"{'-' * 60}")
        print(f"System Efficiency Report (K={self.args.num_clients})")
        print(f" - Max Client Time   : {metrics['max_client_time']:.4f} s")
        print(f" - Server Agg Time   : {metrics['server_agg_time']:.4f} s")
        print(f" - Total Fed Time    : {metrics['total_time']:.4f} s")
        print(f"{'=' * 60}\n")

    def aggregate(self):
        """Aggregates client prototypes using James-Stein shrinkage.

        Returns:
            dict: Latency and computational metrics.
        """
        client_stats = []
        client_times = []

        # 1. Simulate parallel client computation
        for client in self.clients:
            t_start = time.time()
            stats = client.get_stats()
            t_end = time.time()

            client_times.append(t_end - t_start)
            client_stats.append(stats)

        max_client_time = max(client_times)

        # 2. Server-side aggregation
        server_t_start = time.time()

        self.global_prototypes = torch.zeros((self.num_classes, self.feature_dim), device=self.device)
        global_n_c = torch.zeros(self.num_classes, device=self.device)

        for c in range(self.num_classes):
            cids = [i for i, stats in enumerate(client_stats) if c in stats]
            if not cids:
                continue

            mu_list = torch.stack([client_stats[i][c]['mu'] for i in cids]).to(self.device)
            n_c_tensor = torch.tensor([client_stats[i][c]['n_c'] for i in cids], dtype=torch.float32,
                                      device=self.device)

            for idx, cid in enumerate(cids):
                n_c = n_c_tensor[idx]

                # James-Stein shrinkage factor
                lambda_n = 1.0 / (1.0 + self.gamma * n_c)

                mask = torch.ones(len(cids), dtype=torch.bool, device=self.device)
                mask[idx] = False

                mu_i = mu_list[idx]
                mu_peer = torch.mean(mu_list[mask], dim=0) if mask.sum() > 0 else mu_i

                # Shrinkage estimation
                mu_tilde = (1 - lambda_n) * mu_i + lambda_n * mu_peer

                self.global_prototypes[c] += mu_tilde * n_c
                global_n_c[c] += n_c

        # Normalize global prototypes
        valid_classes = global_n_c > 0
        self.global_prototypes[valid_classes] /= global_n_c[valid_classes].unsqueeze(1)
        self.global_prototypes = F.normalize(self.global_prototypes, p=2, dim=1)

        server_t_end = time.time()
        server_agg_time = server_t_end - server_t_start

        return {
            'max_client_time': max_client_time,
            'server_agg_time': server_agg_time,
            'total_time': max_client_time + server_agg_time
        }

    def _ppr_smooth_server(self, x, edge_index, num_nodes=None):
        """Applies PPR smoothing on the central evaluation graph."""
        if num_nodes is None:
            num_nodes = x.size(0)

        x_norm = F.normalize(x, p=2, dim=1)

        if edge_index.size(1) == 0:
            return x_norm

        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop
        deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)

        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row]

        adj_sparse = torch.sparse_coo_tensor(
            edge_index_loop, norm, (num_nodes, num_nodes), device=self.device
        )

        h_k = x_norm.clone()
        for _ in range(self.T):
            h_k = (1 - self.alpha) * x_norm + self.alpha * torch.sparse.mm(adj_sparse, h_k)

        return h_k

    def _infer_with_prototype(self, test_x, prototypes):
        """Performs inference using cosine similarity to class prototypes."""
        test_x_norm = F.normalize(test_x, p=2, dim=1)
        logits = torch.matmul(test_x_norm, prototypes.t())
        return torch.argmax(logits, dim=1)

    def global_evaluate(self):
        """Evaluates the aggregated global prototypes on the test set."""
        eval_data = self.data.to(self.device)
        mask = getattr(eval_data, 'test_mask', None)

        if mask is None or mask.sum().item() == 0:
            return 0.0, 0.0

        y_true = eval_data.y[mask]

        # Extract central smoothed features
        x_smooth_central = self._ppr_smooth_server(eval_data.x, eval_data.edge_index, eval_data.num_nodes)

        # Inference and metrics calculation
        preds = self._infer_with_prototype(x_smooth_central, self.global_prototypes)
        acc = preds[mask].eq(y_true).sum().item() / mask.sum().item()
        f1 = f1_score(y_true.cpu().numpy(), preds[mask].cpu().numpy(), average='macro')

        print(f"[Evaluation] Test Accuracy: {acc * 100:.2f}% | Macro F1: {f1:.4f}")

        return acc, f1