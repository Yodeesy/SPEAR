# algorithm/base.py
import random
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def get_tensor_size_mb(obj):
    """Calculates the size of a Tensor or nn.Module in megabytes (MB).

    Args:
        obj: The object to measure (Tensor, nn.Module, list, tuple, or scalar).

    Returns:
        float: Estimated size in MB.
    """
    if isinstance(obj, torch.Tensor):
        return obj.nelement() * obj.element_size() / (1024 * 1024)
    elif isinstance(obj, torch.nn.Module):
        return sum(p.nelement() * p.element_size() for p in obj.parameters()) / (1024 * 1024)
    elif isinstance(obj, (list, tuple)):
        return sum(get_tensor_size_mb(item) for item in obj)
    elif isinstance(obj, (float, int)):
        return 8 / (1024 * 1024)  # Assume 8 bytes for scalars
    return 0.0


class BaseServer:
    """Base class for the Federated Learning Server.

    Handles client sampling, parameter broadcasting, aggregation, and global evaluation.
    Designed to support both training-based methods (e.g., FedAvg) and
    training-free methods (e.g., SPEAR).
    """

    def __init__(self, args, clients, model, data, logger):
        self.logger = logger
        self.sampled_clients = None
        self.clients = clients
        self.model = model  # Global model (None for training-free methods)
        self.cl_sample_rate = args.cl_sample_rate
        self.num_rounds = args.num_rounds
        self.T_L = args.T_L  # Local training epochs
        self.data = data

        # Determine device from model parameters if available
        self.device = next(model.parameters()).device if model is not None else torch.device("cpu")
        self.num_total_samples = sum([client.num_samples for client in self.clients])

        self.best_val_acc = 0.0
        self.test_acc_at_best_val = 0.0
        self.stop_training = False
        self.total_uplink_mb = 0.0

    def run(self):
        """Main training loop for Federated Learning."""
        for round_idx in range(self.num_rounds):
            print(f"\n--- Round {round_idx + 1} ---")
            self.logger.write_round(round_idx + 1)

            # 1. Client Sampling
            self.sample()

            # 2. Parameter Broadcasting (Only applies to training-based methods)
            self.communicate()

            # 3. Local Training
            for cid in self.sampled_clients:
                client = self.clients[cid]
                # Distinguish between training-based and training-free clients
                if client.model is not None:
                    for _ in range(self.T_L):
                        client.train()

            # 4. Global Aggregation
            # Training-based methods (FedAvg) average parameters here.
            # Training-free methods (SPEAR) override this to aggregate prototypes.
            self.aggregate()

            if self.stop_training:
                print(f"Training stopped early at Round {round_idx + 1}.")
                break

            # 5. Global Evaluation
            val_acc, test_acc, test_f1, loss_val = self.global_evaluate()

            print(f"Result: Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Loss: {loss_val:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.test_acc_at_best_val = test_acc
                print(f" * New Best Validation Accuracy! (Test Acc: {test_acc:.4f})")

        print(f"\n{'=' * 40}")
        print("FEDERATED LEARNING COMPLETED")
        print(f"Final Best Test Accuracy: {self.test_acc_at_best_val:.4f}")
        print(f"{'=' * 40}\n")

    def communicate(self):
        """Broadcasts global model parameters to sampled clients.

        Note:
            Bypassed for training-free methods where `client.model` is None.
        """
        if self.model is None:
            return

        for cid in self.sampled_clients:
            client = self.clients[cid]
            if client.model is not None:
                for client_param, server_param in zip(client.model.parameters(), self.model.parameters()):
                    client_param.data.copy_(server_param.data)

    def sample(self):
        """Randomly samples a fraction of clients for the current round."""
        num_sample_clients = int(len(self.clients) * self.cl_sample_rate)
        num_sample_clients = max(1, num_sample_clients)  # Ensure at least 1 client
        self.sampled_clients = random.sample(range(len(self.clients)), num_sample_clients)

    def aggregate(self):
        """Standard FedAvg aggregation strategy.

        Aggregates parameters from sampled clients weighted by their dataset size.
        Note: Training-free methods like SPEAR will override this method completely.
        """
        if self.model is None:
            return

        total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])

        for i, cid in enumerate(self.sampled_clients):
            client = self.clients[cid]

            if client.model is None:
                continue

            self.total_uplink_mb += get_tensor_size_mb(client.model)
            weight = client.num_samples / total_samples

            for client_param, global_param in zip(client.model.parameters(), self.model.parameters()):
                if i == 0:
                    global_param.data.copy_(weight * client_param)
                else:
                    global_param.data += weight * client_param

    def global_evaluate(self):
        """Evaluates the global model on both Validation and Test sets.

        Returns:
            tuple: (val_acc, test_acc, test_f1, test_loss)
        """
        if self.model is None:
            return 0.0, 0.0, 0.0, 0.0

        self.model.eval()

        # Fallback to CPU for massive graphs to avoid OOM
        eval_device = self.device
        if hasattr(self.data, 'x') and self.data.x.shape[0] > 100000:
            eval_device = torch.device('cpu')
            self.model.to(eval_device)

        eval_data = self.data.to(eval_device) if hasattr(self.data, 'to') else self.data

        val_acc, test_acc, test_f1, loss_val = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            out = self.model(eval_data)
            logits = out[-1] if isinstance(out, (tuple, list)) else out

            # Validation Evaluation
            if hasattr(eval_data, 'val_mask') and eval_data.val_mask is not None and eval_data.val_mask.sum() > 0:
                pred_val = logits[eval_data.val_mask].max(dim=1)[1]
                correct_val = pred_val.eq(eval_data.y[eval_data.val_mask]).sum().item()
                val_acc = correct_val / eval_data.val_mask.sum().item()

            # Test Evaluation
            mask = getattr(eval_data, 'test_mask', None)
            if mask is not None and mask.sum().item() > 0:
                loss = F.cross_entropy(logits[mask], eval_data.y[mask])
                pred_test = logits[mask].max(dim=1)[1]
                test_acc = pred_test.eq(eval_data.y[mask]).sum().item() / mask.sum().item()

                y_true = eval_data.y[mask].cpu().numpy()
                y_pred = pred_test.cpu().numpy()
                test_f1 = f1_score(y_true, y_pred, average='macro')
                loss_val = loss.item()
            else:
                # Inductive fallback (no test mask provided)
                loss = F.cross_entropy(logits, eval_data.y)
                pred_test = logits.max(dim=1)[1]
                test_acc = pred_test.eq(eval_data.y).sum().item() / eval_data.y.shape[0]
                loss_val = loss.item()

            self.logger.write_test_loss(loss_val)
            self.logger.write_test_acc(test_acc)
            if hasattr(self.logger, 'write_test_f1'):
                self.logger.write_test_f1(test_f1)

        if eval_device.type == 'cpu':
            self.model.to(self.device)

        return val_acc, test_acc, test_f1, loss_val


class BaseClient:
    """Base class for a Federated Learning Client.

    Handles local data management and local training operations.
    """

    def __init__(self, args, model, data):
        self.model = model
        self.data = data.clone() if hasattr(data, 'clone') else data
        self.args = args
        self.num_samples = len(data.x) if data.x is not None else 0
        self.loss_fn = F.cross_entropy

        # Initialize optimizer only for training-based methods
        if self.model is not None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        else:
            self.optimizer = None

    def train(self):
        """Performs local training for one epoch.

        Returns:
            float: The training loss for the epoch. Returns 0.0 for training-free methods.
        """
        if self.model is None:
            return 0.0

        device = next(self.model.parameters()).device
        train_data = self.data.to(device)

        if hasattr(self.data, 'edge_index') and self.data.edge_index is not None:
            self.data.edge_index = self.data.edge_index.long()
        train_data.edge_index = train_data.edge_index.long()

        self.model.train()
        self.optimizer.zero_grad()

        _, out = self.model(train_data)

        if hasattr(train_data, 'train_mask') and train_data.train_mask is not None:
            loss = self.loss_fn(out[train_data.train_mask], train_data.y[train_data.train_mask])
        else:
            loss = self.loss_fn(out, train_data.y)

        loss.backward()
        self.optimizer.step()

        return loss.item()