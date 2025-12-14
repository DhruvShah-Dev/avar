import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from avar.fouls.mvfoul_dataset import MVFoulDataset
from avar.fouls.mvfoul_model import SimpleVideoFoulNet


def train(
    meta_train: Path,
    meta_val: Path,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    num_workers: int = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = MVFoulDataset(str(meta_train))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    if meta_val is not None and meta_val.exists():
        val_ds = MVFoulDataset(str(meta_val))
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    model = SimpleVideoFoulNet(num_foul_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device == "cuda"))

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for clips, targets in pbar:
            clips = clips.to(device, non_blocking=True)
            labels = targets["foul"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device == "cuda")):
                logits = model(clips)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * clips.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(
                loss=total_loss / total if total else 0.0,
                acc=correct / total if total else 0.0,
            )

        train_loss = total_loss / total
        train_acc = correct / total

        val_acc = None
        if val_loader is not None:
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.inference_mode():
                for clips, targets in val_loader:
                    clips = clips.to(device, non_blocking=True)
                    labels = targets["foul"].to(device, non_blocking=True)
                    logits = model(clips)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)
            val_acc = v_correct / v_total if v_total else 0.0

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_acc={val_acc:.4f}" if val_acc is not None else ""
        )

        # Save checkpoint
        ckpt_path = out_dir / f"foul_epoch{epoch}.pt"
        torch.save({"model_state": model.state_dict()}, ckpt_path)

        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = out_dir / "foul_best.pt"
            torch.save({"model_state": model.state_dict()}, best_path)

    print("Training complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", type=Path, required=True)
    ap.add_argument("--meta-val", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("./models/fouls"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    train(
        meta_train=args.meta_train,
        meta_val=args.meta_val,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
