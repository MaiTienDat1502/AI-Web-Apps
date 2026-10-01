import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

import core.classifier as clf

from config import ART_DIR, DEVICE


# =========================================================
# CONFIG
# =========================================================

SEED = 42

ROOT = Path(__file__).resolve().parent

FLOWERS_DIR = (
    ROOT
    / "data"
    / "flowers"
    / "flower_photos"
)

MODEL_DIR = ART_DIR / "classifier"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# FIX RANDOM SEED
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


print("========================================")
print(" IMAGE CLASSIFICATION - RESNET18")
print("========================================")
print()

print("Device:", DEVICE)
print("Dataset:", FLOWERS_DIR)
print()


# =========================================================
# KIỂM TRA DATASET
# =========================================================

if not FLOWERS_DIR.exists():

    raise FileNotFoundError(
        "Không tìm thấy dataset TF Flowers.\n"
        "Hãy chạy: python download_flowers.py"
    )


# =========================================================
# LOAD DATASET
# =========================================================

base = ImageFolder(
    FLOWERS_DIR
)

classes = base.classes

targets = np.array(
    base.targets
)

print("Các lớp:")

for name in classes:
    print("-", name)

print()

print(
    "Tổng số ảnh:",
    len(base),
)

print()


# =========================================================
# CHIA TRAIN / VALIDATION / TEST
#
# 80% train
# 10% validation
# 10% test
# =========================================================

all_indices = np.arange(
    len(targets)
)


train_idx, temp_idx = train_test_split(
    all_indices,
    test_size=0.2,
    stratify=targets,
    random_state=SEED,
)


val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=targets[temp_idx],
    random_state=SEED,
)


print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))
print()


# =========================================================
# DATASET VỚI TRANSFORM
# =========================================================

train_ds = Subset(
    ImageFolder(
        FLOWERS_DIR,
        transform=clf.TRAIN_TF,
    ),
    train_idx,
)


val_ds = Subset(
    ImageFolder(
        FLOWERS_DIR,
        transform=clf.EVAL_TF,
    ),
    val_idx,
)


test_ds = Subset(
    ImageFolder(
        FLOWERS_DIR,
        transform=clf.EVAL_TF,
    ),
    test_idx,
)


# =========================================================
# DATALOADER
# =========================================================

num_workers = 0

train_dl = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
)


val_dl = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
)


test_dl = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
)


# =========================================================
# LƯU SPLIT
# =========================================================

split_data = {
    "train": train_idx.tolist(),
    "val": val_idx.tolist(),
    "test": test_idx.tolist(),
}

(
    MODEL_DIR / "split.json"
).write_text(
    json.dumps(
        split_data,
        indent=2,
    ),
    encoding="utf-8",
)


# =========================================================
# TẠO MODEL
# =========================================================

print("Đang tải ResNet-18 ImageNet...")

model = clf.build_model(
    len(classes),
    pretrained=True,
)

model = model.to(DEVICE)

print("Đã tải model.")
print()


# =========================================================
# TRAINING CONFIG
# =========================================================

EPOCHS = 3

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
)


scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=EPOCHS * len(train_dl),
)


use_amp = DEVICE == "cuda"


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

else:

    scaler = None


# =========================================================
# RUN ONE EPOCH
# =========================================================

def run_epoch(
    dataloader,
    train=False,
):

    model.train(train)

    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in dataloader:

        x = x.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        if train:

            optimizer.zero_grad(
                set_to_none=True
            )


        if use_amp:

            with torch.autocast(
                DEVICE,
                dtype=torch.float16,
            ):

                logits = model(x)

                loss = criterion(
                    logits,
                    y,
                )

        else:

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )


        if train:

            if use_amp:

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                optimizer.step()


            scheduler.step()


        loss_sum += (
            loss.item()
            * len(y)
        )

        correct += (
            logits.argmax(1) == y
        ).sum().item()

        total += len(y)


    return (
        loss_sum / total,
        correct / total,
    )


# =========================================================
# TRAIN
# =========================================================

best_acc = 0.0

history = []


for epoch in range(
    1,
    EPOCHS + 1,
):

    start = time.time()

    train_loss, train_acc = run_epoch(
        train_dl,
        train=True,
    )

    val_loss, val_acc = run_epoch(
        val_dl,
        train=False,
    )

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    history.append(
        record
    )

    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"| train loss={train_loss:.3f} "
        f"acc={train_acc:.3f} "
        f"| val loss={val_loss:.3f} "
        f"acc={val_acc:.3f} "
        f"| {time.time() - start:.0f}s"
    )


    # Chỉ lưu model tốt nhất dựa trên validation
    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_DIR / "model.pt",
        )


print()
print(
    "Best validation accuracy:",
    round(best_acc, 4),
)


# =========================================================
# LƯU CLASSES
# =========================================================

(
    MODEL_DIR / "classes.json"
).write_text(
    json.dumps(
        classes,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# =========================================================
# TEST
# =========================================================

print()
print("Đang đánh giá trên TEST...")


model.load_state_dict(
    torch.load(
        MODEL_DIR / "model.pt",
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()


y_true = []
y_pred = []


with torch.inference_mode():

    for x, y in test_dl:

        logits = model(
            x.to(DEVICE)
        )

        predictions = (
            logits.argmax(1)
            .cpu()
            .tolist()
        )

        y_pred.extend(
            predictions
        )

        y_true.extend(
            y.tolist()
        )


test_accuracy = float(
    np.mean(
        np.array(y_true)
        == np.array(y_pred)
    )
)


test_f1 = float(
    f1_score(
        y_true,
        y_pred,
        average="macro",
    )
)


print()

print(
    classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=3,
    )
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {
    "test_accuracy": test_accuracy,
    "test_macro_f1": test_f1,
    "epochs": EPOCHS,
    "history": history,
    "model": "resnet18-imagenet-finetune",
}


(
    MODEL_DIR / "metrics.json"
).write_text(
    json.dumps(
        metrics,
        indent=2,
    ),
    encoding="utf-8",
)


print("========================================")
print("TRAINING HOÀN TẤT")
print("========================================")
print()
print("Accuracy:", round(test_accuracy, 4))
print("Macro F1:", round(test_f1, 4))
print()
print("Model:", MODEL_DIR / "model.pt")
print("Classes:", MODEL_DIR / "classes.json")
print("Metrics:", MODEL_DIR / "metrics.json")