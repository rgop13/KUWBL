# mnist_distributed.py
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# ---- Distributed env ----
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.environ.get("RANK", 0))


# -------- Dataset wrapper (HF Datasets -> PyTorch Dataset) --------
class HFDataset(Dataset):
    """
    Wraps a Hugging Face Dataset split (with features: image, label) and applies torchvision transforms.
    Expects `image` to be PIL.Image or array-like; `label` to be int-like.
    """
    def __init__(self, hf_ds, transform: Optional[transforms.Compose] = None):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]

        # Normalize image object to PIL.Image
        if isinstance(img, Image.Image):
            pil = img
        else:
            arr = np.array(img, dtype=np.uint8)
            if arr.size == 28 * 28:
                arr = arr.reshape(28, 28)
            pil = Image.fromarray(arr, mode="L")

        if self.transform:
            pil = self.transform(pil)

        label = int(ex["label"]) if "label" in ex else int(ex["labels"])
        return pil, label


# ----------------- Model -----------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# --------------- Train/Test ---------------
def train(args, model, device, loader, optimizer, epoch, writer: Optional[SummaryWriter]):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if GLOBAL_RANK == 0 and batch_idx % args.log_interval == 0:
            msg = (
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(loader.dataset) // max(WORLD_SIZE, 1)} "
                f"({100.0 * batch_idx / max(len(loader), 1):.0f}%)]\t"
                f"loss={loss.item():.4f}"
            )
            print(msg)
            if writer is not None:
                niter = (epoch - 1) * len(loader) + batch_idx
                writer.add_scalar("loss", loss.item(), niter)


@torch.no_grad()
def test(args, model, device, loader, writer: Optional[SummaryWriter], epoch: int):
    model.eval()

    local_loss_sum = torch.tensor(0.0, device=device)
    local_num = torch.tensor(0.0, device=device)
    local_correct = torch.tensor(0.0, device=device)

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(data)
        loss = F.nll_loss(output, target, reduction="sum")
        pred = output.argmax(dim=1)
        correct = (pred == target).sum()

        local_loss_sum += loss
        local_correct += correct.float()
        local_num += torch.tensor(float(data.size(0)), device=device)

    # Reduce across workers
    dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_num, op=dist.ReduceOp.SUM)

    # Average loss per sample; accuracy global
    mean_loss = (local_loss_sum / torch.clamp(local_num, min=1.0)).item()
    accuracy = (local_correct / torch.clamp(local_num, min=1.0)).item()

    if GLOBAL_RANK == 0:
        print(f"\nval_loss={mean_loss:.4f}  accuracy={accuracy:.4f}\n")
        if writer is not None:
            writer.add_scalar("val_loss", mean_loss, epoch)
            writer.add_scalar("accuracy", accuracy, epoch)


# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser(description="PyTorch Distributed Fashion-MNIST (HF Datasets)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    ## pytorch 으로 실행 시, arguments로 지정 가능합니다. pytorchjob.yaml 파일의 command 항목 참고하시면 됩니다.
    parser.add_argument("--checkpoint_path", default="/tmp/checkpoints/mnist_distributed.pt")  # 해당 경로 volume으로 구성해서 마운트 필요 (/tmp로 하여 하위 경로 생성)
    parser.add_argument("--log_path", default="/tmp/logs") # 해당 경로 volume으로 구성해서 마운트 필요 (/tmp로 하여 하위 경로 생성)
    parser.add_argument("--data_path", default="/data/test/fashion_mnist", help="root containing dataset_dict.json") # 데이터가 있는 경로입니다. test용 fashion_mnist 경로
    parser.add_argument("--backend", type=str, default="nccl", help="nccl|gloo")

    args = parser.parse_args()

    # Paths
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Init process group
    if GLOBAL_RANK == 0:
        print(f"Using distributed PyTorch with '{args.backend}' backend")
    dist.init_process_group(backend=args.backend)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(LOCAL_RANK)
    else:
        device = torch.device("cpu")
        if GLOBAL_RANK == 0:
            print("WARNING: CUDA not available; using CPU.")

    # Seeds (rank-aware)
    torch.manual_seed(args.seed + GLOBAL_RANK)
    np.random.seed(args.seed + GLOBAL_RANK)

    # SummaryWriter only on rank0
    writer = SummaryWriter(args.log_path) if GLOBAL_RANK == 0 else None

    # Transforms (Fashion-MNIST typical stats)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # ---- Load HF dataset from disk (expects dataset_dict.json at data_path) ----
    if GLOBAL_RANK == 0:
        print(f"Loading HF Dataset from: {args.data_path}")
    ds = load_from_disk(args.data_path)  # must contain train/test splits
    if not all(k in ds for k in ("train", "test")):
        raise ValueError(f"Expected train/test splits at {args.data_path}, got keys: {list(ds.keys())}")

    train_dataset = HFDataset(ds["train"], transform=transform)
    test_dataset = HFDataset(ds["test"], transform=transform)

    if GLOBAL_RANK == 0:
        print(f"Loaded dataset: train={len(train_dataset)} / test={len(test_dataset)}")

    # Samplers/Loaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
    )

    # Model/optim
    model = Net().to(device)
    model = DDP(model, device_ids=[LOCAL_RANK] if device.type == "cuda" else None)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train
    for epoch in range(1, args.epochs + 1):
        # required for DistributedSampler shuffling
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, writer, epoch)

    # Save checkpoint (barrier then rank0 save)
    dist.barrier()
    if GLOBAL_RANK == 0:
        torch.save(model.state_dict(), args.checkpoint_path)
        print(f"Checkpoint saved at {args.checkpoint_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
