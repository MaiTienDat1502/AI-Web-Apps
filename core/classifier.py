"""Ứng dụng 1 — Phân loại ảnh (ResNet-18 fine-tune trên bộ Flowers)."""

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

from config import ART_DIR, DEVICE


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# =========================================================
# TRANSFORM DÙNG KHI TRAIN
# =========================================================

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.7, 1.0),
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(
        0.2,
        0.2,
        0.2,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


# =========================================================
# TRANSFORM DÙNG KHI DỰ ĐOÁN / VALIDATION / TEST
# =========================================================

EVAL_TF = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


# =========================================================
# TẠO RESNET-18
# =========================================================

def build_model(
    num_classes: int,
    pretrained: bool = True,
) -> torch.nn.Module:

    weights = (
        models.ResNet18_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    model = models.resnet18(
        weights=weights
    )

    # Thay lớp cuối của ResNet-18
    # từ 1000 lớp ImageNet thành 5 lớp hoa.
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


# =========================================================
# CLASSIFIER DÙNG CHO API
# =========================================================

class ImageClassifier:

    def __init__(
        self,
        model_dir: Path = ART_DIR / "classifier",
        min_confidence: float = 0.5,
    ):

        model_dir = Path(model_dir)

        classes_file = model_dir / "classes.json"
        model_file = model_dir / "model.pt"

        if not classes_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {classes_file}"
            )

        if not model_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {model_file}"
            )

        self.classes = json.loads(
            classes_file.read_text(
                encoding="utf-8"
            )
        )

        self.model = build_model(
            len(self.classes),
            pretrained=False,
        )

        state = torch.load(
            model_file,
            map_location=DEVICE,
            weights_only=True,
        )

        self.model.load_state_dict(
            state
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.min_confidence = min_confidence


    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> dict:

        image = image.convert("RGB")

        x = EVAL_TF(
            image
        ).unsqueeze(0).to(DEVICE)

        logits = self.model(x)

        probs = logits.softmax(
            dim=-1
        )[0]

        scores, indices = probs.topk(
            min(
                top_k,
                len(self.classes),
            )
        )

        predictions = []

        for score, index in zip(
            scores.tolist(),
            indices.tolist(),
        ):

            predictions.append({
                "label": self.classes[index],
                "score": round(
                    float(score),
                    4,
                ),
            })

        return {
            "predictions": predictions,

            # Nếu độ tin cậy thấp thì báo cho người dùng
            # thay vì khẳng định chắc chắn.
            "confident": (
                predictions[0]["score"]
                >= self.min_confidence
            ),
        }