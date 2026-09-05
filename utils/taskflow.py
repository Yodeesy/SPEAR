# utils/taskflow.py
"""TaskFlow module for orchestrating the Federated Learning pipeline.

This module handles the initialization of datasets, data partitioning,
global and local model instantiation, and the setup of the federated server
and clients.
"""

import time

import numpy as np
import torch

from algorithm import get_server, get_client
from backbone import get_model
from datasets.dataset_ds_loader import partition_by_node
from datasets.dataset_loader import load_dataset
from datasets.partition import dirichlet_partitioner
from utils.logger import DefaultLogger


class TaskFlow:
    """Orchestrates the entire Federated Learning pipeline.

    Attributes:
        args: Command-line arguments and hyperparameters.
        device: The computation device (CPU or GPU).
        dataset: The global dataset loaded for the experiment.
        server: The initialized federated server orchestrating the learning.
    """

    def __init__(self, args):
        """Initializes the TaskFlow pipeline.

        Args:
            args: Parsed command-line arguments.
        """
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

        # Initialize logger
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        exp_name = f"{timestamp}-{args.task}-{args.dataset}-{args.fed_algorithm}"
        logger = DefaultLogger(exp_name, args.logs_dir)

        # ---------------------------------------------------------
        # 1. Load Global Dataset
        # ---------------------------------------------------------
        if args.task == "node_classification":
            self.dataset = load_dataset(args.train_val_test_split, args.dataset_dir, args.dataset, args.seed)

            # Auto-detect dimensions
            input_dim = self.dataset.num_node_features
            num_classes = len(np.unique(self.dataset.y.numpy()))
            args.num_classes = num_classes  # Inject into args for model initialization

            # ---------------------------------------------------------
            # 2. Initialize Global Model (Backbone)
            # ---------------------------------------------------------
            server_model = get_model(
                args.model,
                input_dim,
                args.hidden_dim,
                num_classes,
                args.num_layers,
                args.dropout
            )
            server_model.to(self.device)

            # ---------------------------------------------------------
            # 3. Partition Data (Transductive Setting)
            # ---------------------------------------------------------
            if args.dataset_split_metric == "transductive":
                print("[TaskFlow] Partitioning global graph...")
                # Get node indices per client using Dirichlet distribution
                clients_node_indices = dirichlet_partitioner(
                    self.dataset,
                    args.num_clients,
                    beta=args.dirichlet_beta,
                    least_samples=getattr(args, 'least_samples', 10)
                )

                # Extract subgraphs using the optimized vectorized slicer
                clients_data = partition_by_node(self.dataset, clients_node_indices)

                # ---------------------------------------------------------
                # 4. Initialize Clients
                # ---------------------------------------------------------
                clients = []
                for cid in range(args.num_clients):
                    client_data = clients_data[cid]

                    # Training-free methods (e.g., SPEAR) do not require a local model
                    if args.fed_algorithm == "SPEAR":
                        client_model = None
                    else:
                        # Training-based methods (e.g., FedAvg) require a model copy
                        client_model = get_model(
                            args.model,
                            input_dim,
                            args.hidden_dim,
                            num_classes,
                            args.num_layers,
                            args.dropout
                        )
                        client_model.to(self.device)

                    # Instantiate client
                    client = get_client(args.fed_algorithm, args, client_model, client_data)
                    client.client_id = cid  # Assign ID for logging purposes
                    clients.append(client)

                # ---------------------------------------------------------
                # 5. Initialize Server
                # ---------------------------------------------------------
                # The server receives the full dataset for global transductive evaluation
                server_data = self.dataset

                self.server = get_server(
                    args.fed_algorithm,
                    args,
                    clients,
                    server_model,
                    server_data,
                    logger
                )

            else:
                raise NotImplementedError("The framework currently supports the transductive setting only.")
        else:
            raise NotImplementedError(f"Task '{args.task}' is not supported.")

    def run(self):
        """Executes the federated learning server run loop."""
        print("[TaskFlow] Starting Server Run Loop...")
        self.server.run()