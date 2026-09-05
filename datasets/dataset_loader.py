# datasets/dataset_loader.py

import os
import os.path as osp
from typing import List, Optional, Tuple

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Actor,
    Amazon,
    Coauthor,
    HeterophilousGraphDataset,
    Planetoid,
    WebKB,
    WikiCS,
    WikipediaNetwork,
)


def get_random_split_masks(
    num_nodes: int,
    labels: torch.Tensor,
    train_prop: float = 0.6,
    valid_prop: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates deterministic boolean split masks based on random permutations.

    Args:
        num_nodes: Total number of nodes in the graph.
        labels: Ground truth node labels.
        train_prop: Proportion of nodes used for training.
        valid_prop: Proportion of nodes used for validation.
        seed: Random seed for reproducible permutations.

    Returns:
        A tuple of (train_mask, val_mask, test_mask) as 1D boolean tensors.
    """
    rs = np.random.RandomState(seed)
    perm = torch.as_tensor(rs.permutation(num_nodes))

    train_num = int(num_nodes * train_prop)
    valid_num = int(num_nodes * valid_prop)

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def load_dataset(
    train_val_test_split: Optional[List[float]],
    root_dir: str,
    dataset_name: str,
    seed: int = 42,
) -> Data:
    """Loads a specified graph dataset and enforces standard splits.

    Prioritizes official public/benchmark splits when provided by dataset
    maintainers; falls back to deterministic random splits for datasets without
    official splits.

    Args:
        train_val_test_split: Proportions for train, validation, and test
            splits [train, val, test] if generating random splits. If None,
            defaults to [0.2, 0.4, 0.4].
        root_dir: Root directory where datasets are stored or downloaded.
        dataset_name: Identifier for the target dataset.
        seed: Random seed used for random partition generation.

    Returns:
        A PyTorch Geometric Data object containing graph features, structure,
        labels, and 1D split masks.

    Raises:
        AssertionError: If `dataset_name` is not in the supported set.
    """
    supported_datasets = {
        'cora', 'citeseer', 'pubmed',
        'computers', 'photo',
        'ogbn-arxiv', 'ogbn-products',
        'wikics',
        'coauthor-cs', 'coauthor-physics',
        'chameleon', 'squirrel',
        'texas', 'cornell', 'wisconsin',
        'actor',
        'roman-empire',
    }
    assert dataset_name in supported_datasets, f"Invalid dataset: {dataset_name}"

    if train_val_test_split is None:
        train_val_test_split = [0.2, 0.4, 0.4]

    print(f"[Dataset] Loading {dataset_name} from {root_dir}...")

    # Planetoid datasets (Cora, Citeseer, Pubmed) with public split.
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(
            root=root_dir,
            name=dataset_name,
            split='public',
            transform=T.NormalizeFeatures(),
        )
        return dataset[0]

    # Open Graph Benchmark (OGB) datasets with official splits.
    if dataset_name in ['ogbn-arxiv', 'ogbn-products']:
        dataset = PygNodePropPredDataset(
            name=dataset_name,
            root=root_dir,
            transform=T.ToUndirected(),
        )
        data = dataset[0]
        data.y = data.y.squeeze()

        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True
        return data

    # WikiCS dataset with 20 official splits (defaults to split index 0).
    if dataset_name == 'wikics':
        dataset = WikiCS(
            root=osp.join(root_dir, 'wikics'),
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        return data

    # Wikipedia network datasets using official 10-fold split index 0.
    if dataset_name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(
            root=root_dir,
            name=dataset_name,
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
        return data

    # WebKB datasets using official 10-fold split index 0.
    if dataset_name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(
            root=root_dir,
            name=dataset_name,
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
        return data

    # Actor dataset using official 10-fold split index 0.
    if dataset_name == 'actor':
        dataset = Actor(
            root=osp.join(root_dir, 'actor'),
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
        return data

    # Roman-Empire dataset using official 10-fold split index 0.
    if dataset_name == 'roman-empire':
        dataset = HeterophilousGraphDataset(
            root=osp.join(root_dir, 'roman-empire'),
            name='Roman-Empire',
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
        return data

    # Datasets requiring generated random splits (Amazon, Coauthor).
    if dataset_name in ['computers', 'photo']:
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops(),
        ])
        dataset = Amazon(root=root_dir, name=dataset_name, transform=transform)
        data = dataset[0]
        if data.y.dim() > 1:
            data.y = data.y.squeeze()

    elif dataset_name in ['coauthor-cs', 'coauthor-physics']:
        name_map = {'coauthor-cs': 'CS', 'coauthor-physics': 'Physics'}
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops(),
        ])
        dataset = Coauthor(
            root=root_dir,
            name=name_map[dataset_name],
            transform=transform,
        )
        data = dataset[0]

    # Generate deterministic random split masks for Amazon and Coauthor.
    data.train_mask, data.val_mask, data.test_mask = get_random_split_masks(
        num_nodes=data.num_nodes,
        labels=data.y,
        train_prop=train_val_test_split[0],
        valid_prop=train_val_test_split[1],
        seed=seed,
    )

    return data