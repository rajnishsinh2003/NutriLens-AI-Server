"""
model_utils.py  –  Nutrifexa-AI
Loads the trained EfficientNet-B3 model and runs inference.
"""

import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image

MODEL_PATH  = "model/efficientnet_b3_nutrilens.pth"
CLASS_MAP   = "model/class_indices.json"
IMAGE_SIZE  = 300
DROPOUT_RATE = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.15)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def load_model(model_path: str = MODEL_PATH, class_map_path: str = CLASS_MAP) -> tuple:
    """Load EfficientNet-B3 weights and return (model, idx_to_class)."""
    with open(class_map_path, "r") as f:
        class_to_idx: dict = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    num_classes = len(class_to_idx)
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model, idx_to_class


def predict(image: Image.Image, model: nn.Module, idx_to_class: dict,
            top_k: int = 5) -> list[dict]:
    """
    Run inference on a PIL image.

    Returns a list of dicts sorted by confidence (descending):
        [{"class": str, "confidence": float}, ...]
    """
    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(min(top_k, len(idx_to_class)))

    return [
        {"class": idx_to_class[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_idxs)
    ]
