# dashboard/fl_engine.py
# ------------------------------------------------------------
# Federated Learning engine for Django prototype
# Implements:
#  - Baseline FedAvg (fp32)
#  - Strategy: Top-K sparsification + Quantization (4/8-bit) + Error Feedback (EF)
# using Flower simulation (Ray)
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as transforms

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedProx, FedAdam

from dashboard.job_store import update_job

# ---------------------------
# Repro + Device
# ---------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Model
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10, input_size: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dynamically compute flattened size
        dummy = torch.zeros(1, in_channels, input_size, input_size)
        dummy_out = self.features(dummy)
        flatten_dim = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_one_epoch(model: nn.Module, loader: DataLoader, lr: float = 0.01, proximal_mu: float = 0.0, global_params: Optional[List[np.ndarray]] = None):
    """Train one epoch. If proximal_mu > 0 and global_params provided, adds FedProx proximal term."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        if proximal_mu > 0 and global_params is not None:
            prox_term = 0.0
            for (name, param), g in zip(model.named_parameters(), global_params):
                prox_term += (param - torch.tensor(g, device=param.device, dtype=param.dtype)).pow(2).sum()
            loss = loss + (proximal_mu / 2.0) * prox_term

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------
# Dataset (cached)
# ---------------------------
_TRAINSET = None
_TESTSET = None

# ---------------------------
# Dataset (cached)
# ---------------------------
_DATA_CACHE = {}

def load_dataset_cached(dataset_name: str):

    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        in_channels = 1
        input_size = 28

    elif dataset_name == "fashion":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )

        in_channels = 1
        input_size = 28

    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        in_channels = 3
        input_size = 32

    else:
        raise ValueError("Unknown dataset")

    return trainset, testset, in_channels, input_size


def dirichlet_partition(trainset, num_clients: int, alpha: float = 0.5) -> List[List[int]]:
    """Non-IID split using Dirichlet distribution over class labels."""
    y = np.array(trainset.targets)
    num_classes = 10
    idx_by_class = [np.where(y == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        np.random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]
    class_proportions = np.random.dirichlet([alpha] * num_clients, size=num_classes)

    for c in range(num_classes):
        idxs = idx_by_class[c]
        proportions = class_proportions[c]
        proportions = proportions / proportions.sum()

        split_points = (np.cumsum(proportions) * len(idxs)).astype(int)
        splits = np.split(idxs, split_points[:-1])

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])
    return client_indices


def iid_partition(trainset, num_clients: int) -> List[List[int]]:
    idxs = np.arange(len(trainset))
    np.random.shuffle(idxs)
    splits = np.array_split(idxs, num_clients)
    return [split.tolist() for split in splits]


def make_loaders(trainset, testset, client_indices: List[List[int]], batch_size: int = 64):
    trainloaders = []
    testloader = DataLoader(testset, batch_size=256, shuffle=False)
    for idxs in client_indices:
        subset = Subset(trainset, idxs)
        trainloaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    return trainloaders, testloader


# ---------------------------
# Parameter helpers
# ---------------------------
def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(parameters[i]) for i, k in enumerate(keys)}
    model.load_state_dict(new_state, strict=True)

def params_nbytes(parameters: List[np.ndarray]) -> int:
    return int(sum(p.nbytes for p in parameters))


# ---------------------------
# Quantization + Error Feedback
# ---------------------------
def quantize_minmax(x: np.ndarray, bits: int):
    """
    Min-max uniform quantization into uint8 container.
    bits: 4 or 8
    """
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    x = x.astype(np.float32, copy=False)

    x_min = float(x.min())
    x_max = float(x.max())
    if x_max == x_min:
        q = np.zeros_like(x, dtype=np.uint8)
        return q, x_min, 1.0

    levels = (2 ** bits) - 1
    scale = (x_max - x_min) / levels
    q = np.round((x - x_min) / scale).astype(np.uint8)
    return q, x_min, float(scale)

def dequantize_minmax(q: np.ndarray, x_min: float, scale: float):
    return (q.astype(np.float32) * scale + x_min).astype(np.float32)

def qbytes_count(num_values: int, bits: int) -> int:
    # how many bytes if values are bit-packed
    return int(math.ceil(num_values * bits / 8.0))

class ErrorFeedback:
    def __init__(self):
        self.residuals: Optional[List[np.ndarray]] = None

    def apply(self, update: List[np.ndarray]) -> List[np.ndarray]:
        if self.residuals is None:
            self.residuals = [np.zeros_like(u, dtype=np.float32) for u in update]
        return [(u.astype(np.float32) + r) for u, r in zip(update, self.residuals)]

    def update(self, original: List[np.ndarray], compressed_decompressed: List[np.ndarray]) -> None:
        self.residuals = [(o.astype(np.float32) - c.astype(np.float32))
                          for o, c in zip(original, compressed_decompressed)]


# ---------------------------
# Top-K sparsification (per tensor)
# ---------------------------
def topk_sparsify_tensor(x: np.ndarray, k_frac: float):
    flat = x.reshape(-1).astype(np.float32, copy=False)
    n = flat.size
    k = max(1, int(math.ceil(k_frac * n)))

    idx = np.argpartition(np.abs(flat), -k)[-k:]
    vals = flat[idx]

    order = np.argsort(idx)
    idx = idx[order].astype(np.int32)
    vals = vals[order].astype(np.float32)

    return idx, vals, x.shape

def topk_desparsify_tensor(indices: np.ndarray, values: np.ndarray, shape):
    flat = np.zeros(int(np.prod(shape)), dtype=np.float32)
    flat[indices.astype(np.int32)] = values.astype(np.float32)
    return flat.reshape(shape)

def pack_topk_quant(parameters: List[np.ndarray], k_frac: float, bits: int):
    packed = []
    total_bytes = 0

    for p in parameters:
        p32 = p.astype(np.float32, copy=False)
        idx, vals, shape = topk_sparsify_tensor(p32, k_frac)
        q, vmin, scale = quantize_minmax(vals, bits)

        # transmitted bytes estimate:
        total_bytes += idx.size * 4                # indices int32
        total_bytes += qbytes_count(q.size, bits)  # bit-packed values
        total_bytes += 8                           # vmin + scale float32

        packed.append((idx, q, vmin, scale, shape))

    return packed, total_bytes

def unpack_topk_quant(packed, bits: int) -> List[np.ndarray]:
    params = []
    for (idx, q, vmin, scale, shape) in packed:
        vals = dequantize_minmax(q, vmin, scale)
        dense = topk_desparsify_tensor(idx, vals, shape)
        params.append(dense.astype(np.float32))
    return params


# ---------------------------
# Config
# ---------------------------
@dataclass
class RunConfig:
    lr: float = 0.01
    local_epochs: int = 1
    quant_bits: int | None = None      # None => baseline fp32
    topk_frac: float | None = None     # None => no sparsification
    error_feedback: bool = False
    proximal_mu: float = 0.0           # FedProx; 0 = no proximal term


# ---------------------------
# Flower Client
# ---------------------------
class CommEffClient(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader, cfg, in_channels, input_size):
        self.model = SimpleCNN(
            in_channels=in_channels,
            input_size=input_size
        ).to(DEVICE)

        self.trainloader = trainloader
        self.testloader = testloader
        self.cfg = cfg
        self.ef = ErrorFeedback() if cfg.error_feedback else None

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        download_bytes = params_nbytes(parameters)
        set_parameters(self.model, parameters)

        proximal_mu = float(config.get("proximal_mu", 0.0))
        use_prox = proximal_mu > 0
        global_params = parameters if use_prox else None

        for _ in range(self.cfg.local_epochs):
            train_one_epoch(
                self.model,
                self.trainloader,
                lr=self.cfg.lr,
                proximal_mu=proximal_mu if use_prox else 0.0,
                global_params=global_params,
            )

        new_params = get_parameters(self.model)

        # baseline fp32
        if self.cfg.quant_bits is None and self.cfg.topk_frac is None:
            upload_bytes = params_nbytes(new_params)
            return new_params, len(self.trainloader.dataset), {
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
            }

        # EF before compression
        to_send = new_params
        if self.ef is not None:
            to_send = self.ef.apply(to_send)

        # Strategy: TopK + Quant
        if self.cfg.topk_frac is not None and self.cfg.quant_bits is not None:
            packed, upload_bytes = pack_topk_quant(to_send, self.cfg.topk_frac, self.cfg.quant_bits)
            reconstructed = unpack_topk_quant(packed, self.cfg.quant_bits)

            if self.ef is not None:
                self.ef.update(to_send, reconstructed)

            return reconstructed, len(self.trainloader.dataset), {
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
            }

        raise RuntimeError("Unsupported config. Use baseline OR (topk_frac + quant_bits).")

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


# ---------------------------
# Server strategy + comm logging
# ---------------------------
def weighted_average(metrics):
    total = 0
    sums: Dict[str, float] = {}
    for n, m in metrics:
        total += n
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + n * float(v)
    return {k: v / total for k, v in sums.items()} if total > 0 else {}

_COMM_LOG: List[Dict[str, float]] = []


class FedAvgWithComm(FedAvg):
    def __init__(self, job_id: str | None = None, experiment_phase: str = "baseline", **kwargs):
        super().__init__(**kwargs)
        self.job_id = job_id
        self.experiment_phase = experiment_phase

    def aggregate_fit(self, server_round, results, failures):
        if self.job_id:
            update_job(self.job_id, phase=self.experiment_phase, current_round=server_round, message=f"Aggregating round {server_round}")
        agg = super().aggregate_fit(server_round, results, failures)
        if agg is None:
            return None
        metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        comm = weighted_average(metrics)
        _COMM_LOG.append({"round": server_round, "upload_bytes_avg": comm.get("upload_bytes", 0.0), "download_bytes_avg": comm.get("download_bytes", 0.0)})
        return agg


class FedProxWithComm(FedProx):
    def __init__(self, job_id: str | None = None, experiment_phase: str = "baseline", **kwargs):
        super().__init__(**kwargs)
        self.job_id = job_id
        self.experiment_phase = experiment_phase

    def aggregate_fit(self, server_round, results, failures):
        if self.job_id:
            update_job(self.job_id, phase=self.experiment_phase, current_round=server_round, message=f"Aggregating round {server_round}")
        agg = super().aggregate_fit(server_round, results, failures)
        if agg is None:
            return None
        metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        comm = weighted_average(metrics)
        _COMM_LOG.append({"round": server_round, "upload_bytes_avg": comm.get("upload_bytes", 0.0), "download_bytes_avg": comm.get("download_bytes", 0.0)})
        return agg


class FedAdamWithComm(FedAdam):
    def __init__(self, job_id: str | None = None, experiment_phase: str = "baseline", **kwargs):
        super().__init__(**kwargs)
        self.job_id = job_id
        self.experiment_phase = experiment_phase

    def aggregate_fit(self, server_round, results, failures):
        if self.job_id:
            update_job(self.job_id, phase=self.experiment_phase, current_round=server_round, message=f"Aggregating round {server_round}")
        agg = super().aggregate_fit(server_round, results, failures)
        if agg is None:
            return None
        res, metrics_agg = agg
        metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        comm = weighted_average(metrics)
        _COMM_LOG.append({"round": server_round, "upload_bytes_avg": comm.get("upload_bytes", 0.0), "download_bytes_avg": comm.get("download_bytes", 0.0)})
        return (res, metrics_agg)


# ---------------------------
# Main API for Django view
# ---------------------------
def run_experiment(
    name: str,
    num_clients: int,
    rounds: int,
    fraction_fit: float,
    alpha: float,
    cfg: RunConfig,
    dataset_name: str,
    non_iid: bool = True,
    job_id: str | None = None,
    experiment_phase: str = "baseline",
    strategy_type: str = "fedavg",
) -> pd.DataFrame:
    """
    Returns DataFrame with:
      round, accuracy, upload_mb_avg, download_mb_avg, upload_mb_cum, experiment
    """

    # Optional: prevent Ray from filling /tmp too much (you can change this)
    # Example: store Ray temp inside project folder
    os.environ["RAY_TMPDIR"] = "/tmp/ray"

    # Clean Ray between runs to reduce /tmp growth in repeated web demos
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    global _COMM_LOG
    _COMM_LOG = []

    trainset, testset, in_channels, input_size = load_dataset_cached(dataset_name)

    if non_iid:
        client_idxs = dirichlet_partition(trainset, num_clients=num_clients, alpha=alpha)
    else:
        client_idxs = iid_partition(trainset, num_clients=num_clients)

    trainloaders, testloader = make_loaders(trainset, testset, client_idxs, batch_size=64)

    def client_fn(cid: str):
        return CommEffClient(
            trainloaders[int(cid)],
            testloader,
            cfg,
            in_channels=in_channels,
            input_size=input_size,
        )

    min_fit = max(2, int(num_clients * fraction_fit))
    strat_kw = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=min_fit,
        min_evaluate_clients=min_fit,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    if strategy_type == "fedprox":
        strategy = FedProxWithComm(
            job_id=job_id,
            experiment_phase=experiment_phase,
            proximal_mu=cfg.proximal_mu or 0.01,
            **strat_kw,
        )
    elif strategy_type == "fedadam":
        init_model = SimpleCNN(in_channels=in_channels, input_size=input_size)
        init_ndarrays = get_parameters(init_model)
        init_params = ndarrays_to_parameters(init_ndarrays)
        strategy = FedAdamWithComm(
            job_id=job_id,
            experiment_phase=experiment_phase,
            initial_parameters=init_params,
            **strat_kw,
        )
    else:
        strategy = FedAvgWithComm(
            job_id=job_id,
            experiment_phase=experiment_phase,
            **strat_kw,
        )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        ray_init_args={
            "_temp_dir": "/tmp/ray",
            "include_dashboard": False,
            "ignore_reinit_error": True,
        },
    )

    # Prefer distributed accuracy (client-side evaluate)
    acc_hist = history.metrics_distributed.get("accuracy", [])
    if len(acc_hist) > 0:
        df_acc = pd.DataFrame([{"round": int(rnd), "accuracy": float(val)} for rnd, val in acc_hist])
    else:
        # fallback (should not happen if evaluate_metrics_aggregation_fn is set)
        df_acc = pd.DataFrame([{"round": r, "accuracy": float("nan")} for r in range(1, rounds + 1)])

    df_comm = pd.DataFrame(_COMM_LOG)
    if df_comm.empty:
        df_comm = pd.DataFrame({
            "round": df_acc["round"].tolist(),
            "upload_bytes_avg": [0.0] * len(df_acc),
            "download_bytes_avg": [0.0] * len(df_acc),
        })

    df = df_acc.merge(df_comm, on="round", how="left")
    df["experiment"] = name

    df["upload_mb_avg"] = df["upload_bytes_avg"].fillna(0.0) / (1024 * 1024)
    df["download_mb_avg"] = df["download_bytes_avg"].fillna(0.0) / (1024 * 1024)
    df["upload_mb_cum"] = df["upload_mb_avg"].cumsum()

    return df